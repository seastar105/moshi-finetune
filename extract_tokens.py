import argparse
import gc
import importlib
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import sphn
import submitit
import torch
import torchaudio.functional as F
import whisper_timestamped as whisper
import ray
from audiotools import AudioSignal
from faster_whisper import WhisperModel, BatchedInferencePipeline
import math

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24_000
FRAME_SIZE = 1920       # 80ms * 24kHz -> 1920 samples
CHUNK_SIZE = 100
MIN_DURATION = 2048 / 12.5


def init_logging(verbose: bool = False):
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%m-%d %H:%M:%S",
        force=True,
    )


class AudioReader:
    def __call__(self, item):
        audio_path = item["audio_path"]
        signal = AudioSignal(audio_path)
        if signal.duration < MIN_DURATION or signal.num_channels != 2:
            logger.error(f"File {audio_path} is too short or not stereo. Duration: {signal.duration}, Channels: {signal.num_channels}")
            return []

        if signal.sample_rate != SAMPLE_RATE:
            signal = signal.resample(SAMPLE_RATE)
        
        return [
            {"audio": signal.audio_data[0, 0].squeeze().numpy(), "channel_idx": 0, "audio_path": audio_path},
            {"audio": signal.audio_data[0, 1].squeeze().numpy(), "channel_idx": 1, "audio_path": audio_path},
        ]


class MimiActor:
    def __init__(self):
        from moshi.models import loaders
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
            hf_repo="kyutai/moshiko-pytorch-bf16"
        )
        self.model = checkpoint_info.get_mimi("cuda")
        self.model.eval()
        self.model.torch_compile_encoder_decoder = True
        for param in self.model.parameters():
            param.requires_grad = False

    def __call__(self, item):
        start = time.perf_counter()
        audio = item.pop("audio")
        duration = audio.shape[-1] / SAMPLE_RATE
        code_length = int(math.ceil(duration * 12.5))
        audio = torch.from_numpy(audio).to(device="cuda").unsqueeze(0).unsqueeze(0)
        sample_size = FRAME_SIZE * CHUNK_SIZE
        
        rem = audio.shape[-1] % sample_size
        if rem:
            audio = torch.nn.functional.pad(audio, (0, sample_size - rem), mode="constant", value=0)
        all_codes = []
        
        with torch.no_grad(), self.model.streaming(1):
            for offset in range(0, audio.shape[-1], sample_size):
                frame = audio[..., offset:offset + sample_size]
                codes = self.model.encode(frame)
                all_codes.append(codes.squeeze(0))
        all_codes = torch.cat(all_codes, dim=1).transpose(0, 1)[:code_length, :].cpu().tolist()   #   (T, 8)
        end = time.perf_counter()
        rtf = (end - start) / (audio.shape[-1] / SAMPLE_RATE)
        logger.error(
            f"Processed {item['audio_path']} channel {item['channel_idx']} in {end - start:.2f}s, RTF: {rtf:.4f}"
        )
        return [{
            "audio_path": item["audio_path"],
            "channel_idx": item["channel_idx"],
            "codes": all_codes
        }]
        
                


def main():
    parser = argparse.ArgumentParser(description="Korean STT with Whisper")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    init_logging(verbose=args.debug)
    with open("audios.jsonl") as f:
        data = [json.loads(line) for line in f]
    data = data[args.shard_idx::args.num_shards]
    keep_data = []
    for item in data:
        audio_path = item["audio_path"]
        if not Path(audio_path).with_suffix(".json.code").exists():
            keep_data.append(item)
    data = keep_data
    if args.debug:
        data = data[:200]
    logger.info(f"Processing {len(data)} files.")
    data = ray.data.from_items(data).flat_map(AudioReader, concurrency=(1, 2)).flat_map(MimiActor, concurrency=2, num_gpus=0.5)
    path_to_results = dict()
    for item in data.iter_rows():
        audio_path = item["audio_path"]
        if audio_path not in path_to_results:
            path_to_results[audio_path] = []
        path_to_results[audio_path].append({
            "codes": item["codes"],
            "channel_idx": int(item["channel_idx"]),
        })
        if len(path_to_results[audio_path]) == 2:
            with open(Path(audio_path).with_suffix(".json.code"), "w") as f:
                json.dump(path_to_results[audio_path], f, ensure_ascii=False)
            del path_to_results[audio_path]


if __name__ == "__main__":
    main()

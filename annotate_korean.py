import argparse
import importlib
import json
import logging
import sys
import time
from pathlib import Path

import ray
from audiotools import AudioSignal
from faster_whisper import BatchedInferencePipeline, WhisperModel

transcribe = importlib.import_module("whisper_timestamped.transcribe")
old_get_vad_segments = transcribe.get_vad_segments
logger = logging.getLogger(__name__)

def new_get_vad_segments(*args, **kwargs):
    segs = old_get_vad_segments(*args, **kwargs)
    outs = []
    last_end = 0
    d = int(SAMPLE_RATE * 1.0)
    logger.debug("Reintroducing %d samples at the boundaries of the segments.", d)
    for seg in segs:
        outs.append(
            {"start": max(last_end, seg["start"] - d), "end": seg["end"] + d}
        )
        last_end = outs[-1]["end"]
    return outs

SAMPLE_RATE = 16_000
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


class STTActor:
    def __init__(self, model_name: str = "large-v3-turbo"):
        model = WhisperModel("turbo", device="cuda", compute_type="float16")
        self.model = BatchedInferencePipeline(model=model)

    def __call__(self, item):
        try:
            start = time.perf_counter()
            sample = item.pop("audio")
            # transcribe.get_vad_segments = new_get_vad_segments  # type: ignore
            segments, info = self.model.transcribe(sample, language="ko", word_timestamps=True)
            chunks = []
            for segment in segments:
                if segment.words is None:
                    logger.error("No words in %s: %r", item["audio_path"], segment)
                    continue
                for word in segment.words:
                    try:
                        chunks.append(
                            {"text": word.word, "timestamp": (word.start, word.end)}
                        )
                    except KeyError:
                        logger.error("Missing key in %s: %r", item["audio_path"], word)
                        continue
            result = {
                "audio_path": item["audio_path"],
                "alignments": chunks,
                "channel_idx": item["channel_idx"],
            }
            end = time.perf_counter()
            rtf = (end - start) / (len(sample) / SAMPLE_RATE)
            logger.error(
                f"Processed {item['audio_path']} channel {item['channel_idx']} in {end - start:.2f}s, RTF: {rtf:.4f}"
            )
            return [result]
        except Exception as e:
            logger.error(f"Error processing {item['audio_path']} channel {item['channel_idx']}: {e}")
            return []

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
        if not Path(audio_path).with_suffix(".json").exists():
            keep_data.append(item)
    data = keep_data
    if args.debug:
        data = data[:10]
    logger.info(f"Processing {len(data)} files.")
    data = ray.data.from_items(data).flat_map(AudioReader, concurrency=(1, 2)).flat_map(STTActor, concurrency=2, num_gpus=0.5)
    path_to_results = {}
    for item in data.iter_rows():
        audio_path = item["audio_path"]
        if audio_path not in path_to_results:
            path_to_results[audio_path] = []
        alignments = item["alignments"]
        for alignment in alignments:
            alignment["timestamp"] = (
                float(alignment["timestamp"][0]),
                float(alignment["timestamp"][1]),
            )
        path_to_results[audio_path].append({
            "alignments": item["alignments"],
            "channel_idx": int(item["channel_idx"]),
        })
        if len(path_to_results[audio_path]) == 2:
            with open(Path(audio_path).with_suffix(".json"), "w") as f:
                json.dump(path_to_results[audio_path], f, ensure_ascii=False)
            del path_to_results[audio_path]


if __name__ == "__main__":
    main()

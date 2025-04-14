import dataclasses
import logging
import math
from collections import OrderedDict
from typing import (  # Added List, Tuple, Set
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import fire
import lightning as L
import numpy as np
import safetensors
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.distributed.fsdp.xla_fully_sharded_data_parallel as fsdp_internal
from datasets import load_dataset
from lightning.fabric.accelerators import XLAAccelerator
from lightning.fabric.strategies import XLAFSDPStrategy
from lightning_utilities.core.rank_zero import rank_prefixed_message
from moshi.models import loaders
from moshi.modules.transformer import StreamingTransformerLayer
from torch.nn.utils.rnn import PackedSequence
from torch.optim import AdamW, lr_scheduler

from finetune.args import TrainArgs
from finetune.data.data_loader import Batch
from finetune.loss import compute_loss_with_mask
from finetune.monitoring.utils import set_logger

logger = logging.getLogger("train")


def rank_print(fabric: L.Fabric, message: object, *, flush: bool = True, **kwargs: Any) -> None:
    if fabric.local_rank == 0:
        message = str(message)
        # let each host print, but only on rank 0
        message = rank_prefixed_message(message, fabric.global_rank)
        # TPU VM will only print when the script finishes if `flush=False`
        print(message, flush=flush, **kwargs)

def num_parameters(module: nn.Module, requires_grad: Optional[bool] = None) -> int:
    total = 0
    for p in module.parameters():
        if requires_grad is None or p.requires_grad == requires_grad:
            if hasattr(p, "quant_state"):
                # bitsandbytes 4bit layer support
                total += math.prod(p.quant_state.shape)
            else:
                total += p.numel()
    return total

def patched_apply_to_tensors(
    fn: Callable, container: Union[torch.Tensor, Dict, List, Tuple, Set, Any] # Make container Any for dataclass
) -> Union[torch.Tensor, Dict, List, Tuple, Set, Any]: # Return Any for dataclass
  """Recursively apply to all tensor in different kinds of container types (Patched for Dataclasses)."""

  # Use local name _apply for recursion clarity
  def _apply(
      x: Union[torch.Tensor, Dict, List, Tuple, Set, Any]
  ) -> Union[torch.Tensor, Dict, List, Tuple, Set, Any]:
    if torch.is_tensor(x):
      return fn(x)
    # Check if it's a dataclass instance BEFORE checking dict/list/etc.
    elif dataclasses.is_dataclass(x) and not isinstance(x, type): # Check it's an instance, not the class itself
      # Create a dictionary of changes by recursively applying _apply to each field
      changes = {}
      for field in dataclasses.fields(x):
          original_value = getattr(x, field.name)
          processed_value = _apply(original_value)
          # Only include in changes if the value might have actually changed
          # (e.g., if it was a Tensor processed by fn)
          # This check might be too strict depending on fn, but safer than replacing everything
          if processed_value is not original_value:
              changes[field.name] = processed_value
      # If any field was processed and returned a new value, create a new dataclass instance
      if changes:
        return dataclasses.replace(x, **changes)
      else:
        # If no fields were modified by _apply, return the original instance
        return x
    elif isinstance(x, OrderedDict):
      # Important: Create new instance of the *same* OrderedDict subclass
      od = x.__class__()
      for key, value in x.items():
        od[key] = _apply(value)
      return od
    elif isinstance(x, PackedSequence):
       # PackedSequence data needs careful handling, applying to .data might be needed
       # depending on 'fn'. Original code just returned x, let's keep that
       # unless fn specifically needs to modify PackedSequence internals.
       # If fn modifies tensors, it should likely operate on x.data
       # apply_fn_on_data = _apply(x.data) # Example if fn needed to modify data
       # return PackedSequence(apply_fn_on_data, x.batch_sizes, x.sorted_indices, x.unsorted_indices)
       return x # Keep original behavior for now
    elif isinstance(x, dict):
      # Handle regular dicts
      return {key: _apply(value) for key, value in x.items()}
    elif isinstance(x, list):
      return [_apply(item) for item in x] # Use 'item' for clarity
    elif isinstance(x, tuple):
       # Tuples are immutable, so this creates a new tuple
      return tuple(_apply(item) for item in x) # Use 'item' for clarity
    elif isinstance(x, set):
       # Sets are mutable but order is not guaranteed. Creates a new set.
      return {_apply(item) for item in x} # Use 'item' for clarity
    else:
      # Handle non-container types or types we don't explicitly handle
      return x

  # Start the recursion
  return _apply(container)

# monkey patch to support dataclass outputs
fsdp_internal.apply_to_tensors = patched_apply_to_tensors


class PreComputedDataset:
    def __init__(self, dataset_name: str, max_seq_len: int, seed: int, shuffle: bool = True):
        self.dataset = load_dataset(dataset_name, split="train")
        self.max_seq_len = max_seq_len
        self.epoch = 0
        self.seed = seed
        self.shuffle = shuffle

    def __iter__(self):
        self.epoch += 1
        indices = list(range(len(self.dataset)))
        rng = np.random.default_rng(self.epoch + self.seed)
        if self.shuffle:
            rng.shuffle(indices)
        for idx in indices:
            item = self.dataset[idx]
            ch0 = np.array(item["channel0"])        # (T, 9)
            ch1 = np.array(item["channel1"])        # (T, 9)
            T = ch0.shape[0]
            for offset in range(0, T, self.max_seq_len):
                if offset + self.max_seq_len > T:
                    break
                main_channel = rng.choice([0, 1])
                if main_channel == 0:
                    codes = np.concatenate([ch0[offset : offset + self.max_seq_len], ch1[offset : offset + self.max_seq_len, 1:]], axis=1)
                else:
                    codes = np.concatenate([ch1[offset : offset + self.max_seq_len], ch0[offset : offset + self.max_seq_len, 1:]], axis=1)
                # (B, K, T) shape
                codes = codes.T
                yield torch.from_numpy(codes).long().view(1, -1, self.max_seq_len)

def train(fabric: L.Fabric, args: TrainArgs) -> None:
    fabric.seed_everything(args.seed)
    rank_print(fabric, "Loading and Moshi...")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        hf_repo=args.moshi_paths.hf_repo_id,
        moshi_weights=args.moshi_paths.moshi_path,
        mimi_weights=args.moshi_paths.mimi_path,
        tokenizer=args.moshi_paths.tokenizer_path,
        config_path=args.moshi_paths.config_path,
    )
    with torch.device("meta"):
        model = checkpoint_info.get_moshi(
            device="meta",
            dtype=torch.float32,        # xla fsdp requires parameters to be float32
            load_weight=False,
        )
    moshi_weight = checkpoint_info.moshi_weights
    model_state_dict = safetensors.torch.load_file(moshi_weight)
    for k, v in model_state_dict.items():
        model_state_dict[k] = v.to(torch.float32)
    model.load_state_dict(model_state_dict, strict=True, assign=True)
    model = fabric.setup_module(model)

    # if args.param_dtype == "bfloat16":
    #     model = model.to(torch.bfloat16)

    rank_print(fabric, f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
    rank_print(fabric, f"Number of non-trainable parameters: {num_parameters(model, requires_grad=False):,}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=args.optim.lr,
        betas=(0.9, 0.95),
        eps=1e-08,
        weight_decay=args.optim.weight_decay,
    )
    optimizer = fabric.setup_optimizers(optimizer)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.optim.lr,
        total_steps=args.max_steps,
        pct_start=args.optim.pct_start,
    )
    # 5. Load data loaders
    dataset = PreComputedDataset(
        dataset_name=args.data.train_data,
        seed=args.seed + fabric.global_rank,
        max_seq_len=int(args.duration_sec * 12.5),
        shuffle=args.data.shuffle,
    )
    def wrap_dataset(dataset, batch_size):
        while True:
            iterator = iter(dataset)
            sample_list = []
            for sample in iterator:
                sample_list.append(sample)
                if len(sample_list) == batch_size:
                    codes = torch.cat(sample_list, dim=0)
                    batch = Batch(
                        codes=codes,
                        condition_attributes=None,
                    )
                    yield batch
                    sample_list = []

    data_loader = wrap_dataset(dataset, args.batch_size)
    fabric.seed_everything(998244353 + fabric.global_rank)
    model.train()
    xm.mark_step()

    for step in range(args.max_steps):
        batch = next(data_loader)
        codes = fabric.to_device(batch.codes)
        output = model(codes=codes, condition_tensors=None)
        # xm.mark_step()

        text_loss = compute_loss_with_mask(
            output.text_logits,
            codes[:, : model.audio_offset],
            output.text_mask,
            mode="text",
            text_padding_weight=args.text_padding_weight,
            text_padding_ids={
                model.text_padding_token_id,
                model.end_of_text_padding_id,
            },
        )
        audio_loss = compute_loss_with_mask(
            output.logits,
            codes[:, model.audio_offset : model.audio_offset + model.dep_q],
            output.mask,
            mode="audio",
            first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
        )
        mb_loss = text_loss + audio_loss
        # xm.mark_step()
        fabric.backward(mb_loss)

        fabric.clip_gradients(model, optimizer, max_norm=args.max_norm)
        optimizer.step()
        optimizer.zero_grad()
        last_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        xm.mark_step()

        loss_item = mb_loss.detach()
        text_loss_item = text_loss.detach()
        audio_loss_item = audio_loss.detach()

        avg_loss_item = xm.all_reduce("sum", loss_item).item() / fabric.world_size
        avg_text_loss_item = xm.all_reduce("sum", text_loss_item).item() / fabric.world_size
        avg_audio_loss_item = xm.all_reduce("sum", audio_loss_item).item() / fabric.world_size

        rank_print(fabric, f"Step {step + 1}: loss: {avg_loss_item:.4f}, text_loss: {avg_text_loss_item:.4f}, audio_loss: {avg_audio_loss_item:.4f}, lr: {last_lr:.6e}")


def main(config: str):
    args: TrainArgs = TrainArgs.load(config, drop_extra_fields=False)
    set_logger(logging.INFO)
    devices = XLAAccelerator.auto_device_count()
    print(f"Using {devices} device(s)")
    if devices > 1:
        strategy = XLAFSDPStrategy(
            auto_wrap_policy={StreamingTransformerLayer},
            activation_checkpointing_policy={StreamingTransformerLayer},
            state_dict_type="sharded",  # change to "sharded" in multi-host environments where the filesystem is not shared
            sequential_save=True,
            # buffer_dtype=torch.float32,
            # fp32_reduce_scatter=True,
        )
    else:
        strategy = "auto"
    precision = "32-true"
    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision)
    fabric.launch(train, args)


if __name__ == "__main__":
    fire.Fire(main)

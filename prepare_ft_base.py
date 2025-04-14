import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from moshi.models import loaders
from moshi.models.lm_utils import ScaledEmbedding
from safetensors.torch import save_file
from tqdm.auto import tqdm
from transformers import AutoTokenizer


@torch.no_grad()
def init_embedding_module(
    orig_emb: ScaledEmbedding,
    new_emb: ScaledEmbedding,
) -> ScaledEmbedding:
    # Initialize the embedding module with a Gaussian distribution
    dtype = orig_emb.weight.dtype
    emb_weights = orig_emb.weight.data.cuda().to(torch.float32)
    mean = emb_weights.mean(dim=0).cuda()
    orig_vocab_size = new_emb.weight.size()[0]
    sigma = ((emb_weights - mean).T @ (emb_weights - mean)) / orig_vocab_size
    dist = torch.distributions.multivariate_normal.MultivariateNormal(
        mean.to(torch.float32),
        covariance_matrix=1e-5 * sigma.to(torch.float32),
    )  # MultivariateNormal is supported only for float32

    new_vocab_size = new_emb.weight.size()[0]
    new_emb_weights = torch.stack(
        tuple(dist.sample() for _ in tqdm(range(new_vocab_size), desc="Reinitializing embeddings")),
        dim=0,
    ).to(dtype)
    new_emb.weight.data = new_emb_weights.cpu()
    return new_emb


@torch.no_grad()
def init_linear_module(
    orig_linear: nn.Linear,
    new_linear: nn.Linear,
):
    # Initialize the linear module with a Gaussian distribution
    dtype = orig_linear.weight.dtype
    linear_weights = orig_linear.weight.data.cuda().to(torch.float32)
    mean = linear_weights.mean(dim=0).cuda()
    orig_out_features = new_linear.weight.size()[0]
    sigma = ((linear_weights - mean).T @ (linear_weights - mean)) / orig_out_features
    dist = torch.distributions.multivariate_normal.MultivariateNormal(
        mean.to(torch.float32),
        covariance_matrix=1e-5 * sigma.to(torch.float32),
    )  # MultivariateNormal is supported only for float32

    new_out_features = new_linear.weight.size()[0]
    new_linear_weights = torch.stack(
        tuple(dist.sample() for _ in tqdm(range(new_out_features), desc="Reinitializing linear weights")),
        dim=0,
    ).to(dtype)
    new_linear.weight.data = new_linear_weights
    return new_linear


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--text_padding", type=int, required=True)
    parser.add_argument("--end_of_text_padding", type=int, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16")

    args = parser.parse_args()

    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        hf_repo="kyutai/moshiko-pytorch-bf16",
        moshi_weights=None,
        mimi_weights=None,
        tokenizer=None,
        config_path=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    moshi = checkpoint_info.get_moshi(device="cpu")
    norm_emb = moshi.text_emb.norm is not None

    # Reinit text embeddings
    EmbeddingFactory = partial(
        ScaledEmbedding,
        norm=norm_emb,
        zero_idx=moshi.zero_token_id,
    )

    text_emb_dim = moshi.text_emb.weight.shape[1]
    new_text_emb = EmbeddingFactory(vocab_size + 1, text_emb_dim)
    new_text_emb = init_embedding_module(moshi.text_emb, new_text_emb)
    moshi.text_emb = new_text_emb
    print(f"Text Embedding: {moshi.text_emb.weight.shape}")

    # Reinit depformer text embeddings
    depformer_text_emb_dim = moshi.depformer_text_emb.weight.shape[1]
    new_depformer_text_emb = EmbeddingFactory(
        vocab_size + 1,
        depformer_text_emb_dim,
    )
    new_depformer_text_emb = init_embedding_module(
        moshi.depformer_text_emb,
        new_depformer_text_emb,
    )
    moshi.depformer_text_emb = new_depformer_text_emb
    print(f"Depformer Text Embedding: {moshi.depformer_text_emb.weight.shape}")

    # Reinit text head
    orig_head = moshi.text_linear
    use_bias = orig_head.bias is not None
    assert not use_bias, "Bias is not supported yet"
    new_text_head = nn.Linear(
        text_emb_dim,
        vocab_size,
        bias=use_bias,
    )
    moshi.text_linear = init_linear_module(orig_head, new_text_head)
    print(f"Text Head: {moshi.text_linear.weight.shape}")

    dtype = getattr(torch, args.dtype)
    moshi = moshi.to(device="cpu", dtype=dtype)

    # Save the model
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    save_file(moshi.state_dict(), Path(args.out_dir) / "model.safetensors")
    tokenizer.save_pretrained(args.out_dir)
    lm_config = loaders._lm_kwargs

    lm_config["text_card"] = vocab_size
    lm_config["existing_text_padding_id"] = args.text_padding
    lm_config["existing_text_end_padding_id"] = args.end_of_text_padding

    with open(Path(args.out_dir) / "config.json", "w") as f:
        json.dump(lm_config, f, indent=4)

import os
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from util.globals import *
from util.nethook import Trace, set_requires_grad
from util.runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally, save_cached_state

from rome.tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}


def main():
    """
    Command-line utility to precompute cached stats.
    """
    import argparse

    parser = argparse.ArgumentParser(description="ROME_ATTN Statistics Collector")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa("--model_path", default="../../ptms/")
    aa("--model_name", default="EleutherAI/gpt-j-6B", choices=["gpt2-xl", "EleutherAI/gpt-j-6B"])
    aa("--dataset", default="wikipedia", choices=["wikitext", "wikipedia"])
    aa("--layers", default=[8], type=lambda x: list(map(int, x.split(","))))
    aa("--to_collect", default=["muiltmean"], type=lambda x: x.split(","))
    aa("--sample_size", default=100, type=lambda x: None if x == "all" else int(x))
    aa("--batch_tokens", default=10000, type=lambda x: None if x == "any" else int(x))
    aa("--precision", default="float32", choices=["float64", "float32", "float16"])
    aa("--stats_dir", default=STATS_DIR)
    aa("--download", default=1, type=int, choices=[0, 1])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path + args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_path + args.model_name).eval().cuda()
    set_requires_grad(False, model)

    for layer_num in args.layers:
        print(
            f"Computing stats for layer {layer_num} of {args.model_name} "
            f'over {args.sample_size or "all"} samples of {args.dataset}. '
            "Note, the statistics are collected over the inputs to the second MLP layer, "
            "or equivalently the outputs of the first MLP layer."
        )
        for layer_name in [f"transformer.h.{layer_num}.attn.out_proj", f"transformer.h.{layer_num}.mlp.fc_out"]:
            layer_stats(
                model,
                tokenizer,
                layer_name,
                args.stats_dir,
                args.dataset,
                args.to_collect,
                sample_size=args.sample_size,
                precision=args.precision,
                batch_tokens=args.batch_tokens,
                download=args.download,
            )


def layer_stats(
    model,
    tokenizer,
    layer_name,
    stats_dir,
    ds_name,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=True,
    progress=tqdm,
    force_recompute=False
):
    """
    Function to load or compute cached stats.
    """

    def get_ds():
        raw_ds = load_dataset(
            "./caches/wikipedia.py",
            dict(wikitext="wikitext-103-raw-v1", wikipedia="20220301.en")[ds_name],
            cache_dir="./caches"
        )
        try:
            maxlen = model.config.n_positions
        except:
            maxlen = model.config.max_position_embeddings
        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    # Continue with computation of statistics
    batch_size = 100  # Examine this many dataset texts at once
    try:
        npos = model.config.n_positions
    except:
        npos = model.config.max_position_embeddings
    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = "_t{batch_tokens}" + size_suffix
    if model_name is None:
        model_name = model.config._name_or_path.replace("/", "_")

    stats_dir = Path(stats_dir)
    file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    filename = stats_dir / file_extension

    if not filename.exists() and download:
        remote_url = f"{REMOTE_ROOT_URL}/data/stats/{file_extension}"
        try:
            print(f"Attempting to download {file_extension} from {remote_url}.")
            (stats_dir / "/".join(file_extension.split("/")[:-1])).mkdir(
                exist_ok=True, parents=True
            )
            torch.hub.download_url_to_file(remote_url, filename)
            print("Successfully downloaded.")
        except Exception as e:
            print(f"Unable to download due to {e}. Computing locally....")

    #compute cached stats
    try:
        ds = get_ds() if not filename.exists() else None
    except:
        print("get_ds failed, try again")
        ds = get_ds() if not filename.exists() else None

    if progress is None:
        progress = lambda x: x

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    with torch.no_grad():
        if not filename.exists():
            for batch_group in progress(loader, total=batch_count):
                for batch in batch_group:
                    batch = dict_to_(batch, "cuda")
                    with Trace(
                        model, layer_name, retain_input=True, retain_output=False, stop=True
                    ) as tr:
                        if "neox" in model.config._name_or_path:
                            del batch['position_ids']
                        model(**batch)
                    feats = flatten_masked_batch(tr.input, batch["attention_mask"])
                    # feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                    feats = feats.to(dtype=dtype)
                    stat.add(feats)
    return stat


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer

from lgd_bert.data import load_glue_data
from lgd_bert.model_utils import classifier_query, load_tokenizer_and_model
from lgd_bert.samplers import LGDBatchSampler, RandomBatchSampler
from lgd_bert.train import build_lsh_state, coverage_row, set_seed


ROOT = Path(__file__).resolve().parents[1]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def audit_one(args, task: str, variant: str, output_root: Path) -> list[dict]:
    set_seed(args.seed)
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    data = load_glue_data(task, tokenizer, args.max_seq_length)
    train_size = len(data.train)
    steps_per_epoch = math.ceil(train_size / args.batch_size)
    total_steps = args.epochs * steps_per_epoch

    run_dir = output_root / f"{task}_{variant}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "task": task,
        "variant": variant,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "train_size": train_size,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "audit_mode": "sampler_only_static_initial_query_no_backprop",
        "note": "This audits sampler coverage over a full 3-epoch draw schedule. It does not update BERT/classifier weights.",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (run_dir / "data_report.json").write_text(json.dumps(data.report, indent=2, sort_keys=True) + "\n")

    if variant == "random":
        sampler = RandomBatchSampler(train_size, seed=args.seed + 101)
        query = None
    else:
        _tokenizer, model = load_tokenizer_and_model(args.model_name)
        model.to(device)
        labels_np = data.train.labels.numpy().astype("int64")
        lsh_args = SimpleNamespace(
            batch_size=args.batch_size,
            variant=variant,
            include_bias_in_lsh=False,
            lsh_k=args.lsh_k,
            lsh_l=args.lsh_l,
            seed=args.seed,
            probability_eps=args.probability_eps,
        )
        _cls_reps, stored_vectors, lsh, refresh_time = build_lsh_state(lsh_args, model, data.train, labels_np, device)
        sampler = LGDBatchSampler(lsh, stored_vectors, train_size, seed=args.seed + 101, probability_eps=args.probability_eps)
        query = classifier_query(model, variant, include_bias=False)
        (run_dir / "initial_lsh_summary.json").write_text(
            json.dumps({**lsh.summary().__dict__, "refresh_time_sec": refresh_time}, indent=2, sort_keys=True) + "\n"
        )

    rows: list[dict] = []
    counts: Counter[int] = Counter()
    fallback_count = 0
    draw_count = 0
    for step in range(1, total_steps + 1):
        if variant == "random":
            sample = sampler.sample_batch(args.batch_size)
        else:
            sample = sampler.sample_batch(args.batch_size, query)
        counts.update(int(index) for index in sample.indices)
        fallback_count += sum(1 for flag in sample.fallback_flags if flag)
        draw_count += len(sample.indices)
        if step % steps_per_epoch == 0 or step == total_steps:
            epoch_index = len(rows)
            row = coverage_row(epoch_index, counts, train_size, steps_per_epoch, args.batch_size)
            row.update(
                {
                    "task": task,
                    "variant": variant,
                    "seed": args.seed,
                    "fallback_draw_count": fallback_count,
                    "fallback_draw_frac": fallback_count / max(draw_count, 1),
                }
            )
            rows.append(row)
            counts = Counter()
            fallback_count = 0
            draw_count = 0
    write_csv(run_dir / "sample_coverage_by_epoch.csv", rows)
    return rows


def write_report(rows: list[dict], output_root: Path) -> None:
    report = output_root / "COVERAGE_AUDIT_REPORT.md"
    lines = [
        "# Sample Coverage Audit",
        "",
        "Audit mode: full 3-epoch sampler draw schedule, no BERT/classifier weight updates, no W&B.",
        "",
        "| Task | Variant | Epoch | Train N | Draws | Unique seen | Repeated samples | Duplicate draws | Never seen | Max times seen | Fallback frac |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {task} | {variant} | {epoch} | {train_size} | {draw_count} | {unique_seen_count} | "
            "{repeated_sample_count} | {duplicate_draw_count} | {never_seen_count} | {max_times_seen} | {fallback_draw_frac:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Definitions:",
            "",
            "- `Repeated samples`: number of distinct training examples sampled at least twice within that epoch.",
            "- `Duplicate draws`: total extra uses beyond the first use, i.e. `draw_count - unique_seen_count`.",
            "- `Never seen`: training examples with zero draws within that epoch.",
            "",
        ]
    )
    report.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit sampler replacement coverage over full epoch draw schedules.")
    parser.add_argument("--tasks", default="mrpc,rte")
    parser.add_argument("--variants", default="random,paper_lgd,label_aware_lgd")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--lsh_k", type=int, default=7)
    parser.add_argument("--lsh_l", type=int, default=10)
    parser.add_argument("--probability_eps", type=float, default=1e-12)
    parser.add_argument("--output_root", default=None)
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", "/home2/nilan/.cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/home2/nilan/.cache/huggingface/datasets")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else ROOT / "runs" / f"coverage_audit_sampler_only_{stamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for task in [item.strip() for item in args.tasks.split(",") if item.strip()]:
        for variant in [item.strip() for item in args.variants.split(",") if item.strip()]:
            rows = audit_one(args, task, variant, output_root)
            all_rows.extend(rows)
    write_csv(output_root / "sample_coverage_summary.csv", all_rows)
    write_report(all_rows, output_root)
    print(output_root)
    print(output_root / "sample_coverage_summary.csv")
    print(output_root / "COVERAGE_AUDIT_REPORT.md")


if __name__ == "__main__":
    main()


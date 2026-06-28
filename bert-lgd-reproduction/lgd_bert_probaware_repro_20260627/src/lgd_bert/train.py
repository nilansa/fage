from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import DEFAULT_WANDB_PROJECT, ROOT, TrainConfig
from .data import load_glue_data, save_data_report
from .eval import evaluate
from .lsh import SimHashLSH
from .model_utils import (
    build_stored_vectors,
    classifier_head,
    classifier_query,
    compute_cls_representations,
    load_tokenizer_and_model,
    model_state_hash,
)
from .probability import corrected_loss_weights
from .samplers import LGDBatchSampler, RandomBatchSampler
from .wandb_utils import init_wandb, wandb_finish, wandb_log


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def make_run_dir(args) -> Path:
    output_root = Path(args.output_root) if args.output_root else ROOT / "runs"
    output_root.mkdir(parents=True, exist_ok=True)
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = args.run_name or f"{args.task}_{args.variant}_{args.correction}_seed{args.seed}_{stamp}"
        run_dir = output_root / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_csv_row(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_lsh_state(args, model, train_data, labels_np: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray, SimHashLSH, float]:
    start = time.time()
    cls_reps = compute_cls_representations(model, train_data, args.batch_size, device)
    stored_vectors = build_stored_vectors(
        cls_reps,
        labels_np,
        variant=args.variant,
        include_bias=args.include_bias_in_lsh,
    )
    lsh = SimHashLSH(
        dim=stored_vectors.shape[1],
        k=args.lsh_k,
        l=args.lsh_l,
        seed=args.seed + 17,
    )
    lsh.fit(stored_vectors)
    return cls_reps, stored_vectors, lsh, time.time() - start


def should_refresh(args, step: int, steps_per_epoch: int) -> bool:
    if args.variant == "random" or args.refresh_lsh in {"no_refresh", "initial_only"}:
        return False
    if args.refresh_lsh == "every_epoch":
        return step > 0 and steps_per_epoch > 0 and step % steps_per_epoch == 0
    if args.refresh_lsh == "every_n_steps":
        return step > 0 and step % args.refresh_steps == 0
    raise ValueError(f"Unsupported refresh_lsh mode: {args.refresh_lsh}")


def coverage_row(epoch_index: int, counts: Counter[int], train_size: int, steps_in_epoch: int, batch_size: int) -> dict:
    seen_counts = list(counts.values())
    unique_seen = len(seen_counts)
    repeated_sample_count = sum(1 for value in seen_counts if value > 1)
    duplicate_draw_count = sum(value - 1 for value in seen_counts if value > 1)
    never_seen_count = train_size - unique_seen
    return {
        "epoch": epoch_index,
        "train_size": train_size,
        "steps_in_epoch": steps_in_epoch,
        "batch_size": batch_size,
        "draw_count": sum(seen_counts),
        "unique_seen_count": unique_seen,
        "never_seen_count": never_seen_count,
        "never_seen_frac": never_seen_count / train_size,
        "repeated_sample_count": repeated_sample_count,
        "repeated_sample_frac": repeated_sample_count / train_size,
        "duplicate_draw_count": duplicate_draw_count,
        "duplicate_draw_frac_of_draws": duplicate_draw_count / max(sum(seen_counts), 1),
        "max_times_seen": max(seen_counts) if seen_counts else 0,
    }


def train_main(args) -> dict:
    set_seed(args.seed)
    os.environ.setdefault("HF_HOME", "/ssd_scratch/nilan/.cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/ssd_scratch/nilan/.cache/huggingface/datasets")
    run_dir = make_run_dir(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_tokenizer_and_model(args.model_name)
    model.to(device)
    if args.freeze_bert:
        base = model.bert if hasattr(model, "bert") else model.base_model
        for param in base.parameters():
            param.requires_grad = False

    weight, _bias = classifier_head(model)
    classifier_shape = list(weight.shape)
    data = load_glue_data(args.task, tokenizer, args.max_seq_length)
    save_data_report(data.report, run_dir / "data_report.json")
    labels_np = data.train.labels.numpy().astype("int64")
    steps_per_epoch = math.ceil(len(data.train) / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps is not None:
        total_steps = min(total_steps, int(args.max_steps))

    config = TrainConfig(**{key: getattr(args, key) for key in TrainConfig().__dict__ if hasattr(args, key)}).to_dict()
    config.update(
        {
            "run_dir": str(run_dir),
            "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_device_count": torch.cuda.device_count(),
            "torch_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "classifier_weight_shape": classifier_shape,
            "train_size_N": len(data.train),
            "validation_size": len(data.validation),
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "git_commit": git_commit(),
            "exact_command": " ".join(["python"] + os.sys.argv),
            "uniform_correction_check": "If p_i=1/N then 1/(N*p_i)=1 and corrected CE equals mean CE.",
        }
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    run = init_wandb(config, run_dir)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    if args.variant == "random":
        sampler = RandomBatchSampler(len(data.train), seed=args.seed + 101)
        stored_vectors = None
        lsh = None
        refresh_count = 0
        initial_refresh_time = 0.0
    else:
        _cls_reps, stored_vectors, lsh, initial_refresh_time = build_lsh_state(args, model, data.train, labels_np, device)
        sampler = LGDBatchSampler(lsh, stored_vectors, len(data.train), seed=args.seed + 101, probability_eps=args.probability_eps)
        refresh_count = 1
        summary = lsh.summary().__dict__
        (run_dir / "initial_lsh_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    start_time = time.time()
    last_eval = None
    status = "finished"
    error_message = None
    epoch_counts: Counter[int] = Counter()
    coverage_rows: list[dict] = []

    try:
        for step in range(1, total_steps + 1):
            step_start = time.time()
            refresh_time = 0.0
            if should_refresh(args, step - 1, steps_per_epoch):
                _cls_reps, stored_vectors, lsh, refresh_time = build_lsh_state(args, model, data.train, labels_np, device)
                sampler.update_index(lsh, stored_vectors)
                refresh_count += 1

            epoch = (step - 1) / max(steps_per_epoch, 1)
            if args.variant == "random":
                sample = sampler.sample_batch(args.batch_size)
                query_norm = 0.0
            else:
                query = classifier_query(model, args.variant, include_bias=args.include_bias_in_lsh)
                query_norm = float(np.linalg.norm(query))
                sample = sampler.sample_batch(args.batch_size, query)
            if args.audit_sample_coverage:
                epoch_counts.update(int(index) for index in sample.indices)

            batch = data.train.batch(sample.indices, device)
            labels = batch.pop("labels")
            outputs = model(**batch)
            logits = outputs.logits
            losses = F.cross_entropy(logits, labels, reduction="none")
            probabilities = torch.tensor(sample.probabilities, dtype=torch.float32, device=device)
            weight_stats = corrected_loss_weights(
                probabilities,
                train_size=len(data.train),
                mode=args.correction,
                max_weight=args.correction_max_weight,
                eps=args.probability_eps,
            )
            loss = (weight_stats.weights * losses).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}: {float(loss.detach().cpu())}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.max_grad_norm and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            wall = time.time() - start_time
            step_time = time.time() - step_start
            train_metrics = {
                "global_step": step,
                "epoch": epoch,
                "train/loss": float(loss.detach().cpu()),
                "time/wall_clock_sec": wall,
                "time/step_time_sec": step_time,
                "sampler/query_norm": query_norm,
                "correction/weight_mean": weight_stats.weight_mean,
                "correction/weight_min": weight_stats.weight_min,
                "correction/weight_max": weight_stats.weight_max,
                "correction/weight_clipped_frac": weight_stats.weight_clipped_frac,
                "lsh/refresh_count": refresh_count,
                "lsh/refresh_time_sec": refresh_time,
            }
            train_metrics.update(sample.stats())
            if step == 1 and initial_refresh_time:
                train_metrics["lsh/initial_refresh_time_sec"] = initial_refresh_time
            if step % args.log_every == 0:
                wandb_log(run, train_metrics)
                write_csv_row(run_dir / "train_log.csv", train_metrics)

            do_eval = step % args.eval_every == 0 or step == total_steps
            if do_eval:
                eval_metrics = evaluate(model, data.validation, args.batch_size, device)
                last_eval = {
                    "global_step": step,
                    "epoch": epoch,
                    **eval_metrics,
                    "paper_fig5/test_accuracy": eval_metrics["eval/accuracy"],
                    "paper_fig5/test_loss": eval_metrics["eval/loss"],
                    "time/wall_clock_sec": wall,
                }
                wandb_log(run, last_eval)
                write_csv_row(run_dir / "eval_log.csv", last_eval)
            if args.audit_sample_coverage and (step % steps_per_epoch == 0 or step == total_steps):
                epoch_index = len(coverage_rows)
                coverage = coverage_row(
                    epoch_index=epoch_index,
                    counts=epoch_counts,
                    train_size=len(data.train),
                    steps_in_epoch=steps_per_epoch,
                    batch_size=args.batch_size,
                )
                coverage_rows.append(coverage)
                write_csv_row(run_dir / "sample_coverage_by_epoch.csv", coverage)
                epoch_counts = Counter()
    except Exception as exc:
        status = "failed"
        error_message = repr(exc)
        raise
    finally:
        summary = {
            "status": status,
            "error_message": error_message,
            "run_dir": str(run_dir),
            "wandb_url": (run.url if run is not None else None),
            "final_eval": last_eval,
            "refresh_count": refresh_count,
            "model_state_hash": model_state_hash(model),
            "sample_coverage_by_epoch": coverage_rows,
        }
        (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        wandb_finish(run)
    return summary


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train BERT GLUE with random or LGD sampling.")
    parser.add_argument("--task", choices=["mrpc", "rte"], default="mrpc")
    parser.add_argument("--variant", choices=["random", "paper_lgd", "label_aware_lgd"], default="random")
    parser.add_argument("--correction", choices=["none", "full", "clipped", "sqrt", "normalized"], default="none")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--optimizer", choices=["adam"], default="adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--eval_every", type=int, default=25)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--lsh_k", type=int, default=7)
    parser.add_argument("--lsh_l", type=int, default=10)
    parser.add_argument("--refresh_lsh", choices=["initial_only", "every_epoch", "every_n_steps", "no_refresh"], default="every_epoch")
    parser.add_argument("--refresh_steps", type=int, default=100)
    parser.add_argument("--representation", choices=["cls_last_hidden_state"], default="cls_last_hidden_state")
    parser.add_argument("--include_bias_in_lsh", action="store_true")
    parser.add_argument("--correction_max_weight", type=float, default=10.0)
    parser.add_argument("--probability_eps", type=float, default=1e-12)
    parser.add_argument("--freeze_bert", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=0.0)
    parser.add_argument("--wandb_project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--audit_sample_coverage", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = train_main(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

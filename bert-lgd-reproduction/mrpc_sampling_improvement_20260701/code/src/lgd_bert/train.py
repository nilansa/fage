from __future__ import annotations

import argparse
import csv
import gzip
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
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = args.run_name or f"{args.task}_{args.variant}_{args.correction}_seed{args.seed}_{stamp}"
        run_dir = output_root / name
        run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_csv_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if exists:
        with path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            if any(key not in fieldnames for key in row):
                rows = list(reader)
                fieldnames.extend(key for key in row if key not in fieldnames)
                rows.append(row)
                with path.open("w", newline="") as out:
                    writer = csv.DictWriter(out, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                return
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()) if not exists else fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def representation_drift_stats(
    old_cls_reps: np.ndarray | None,
    new_cls_reps: np.ndarray,
    old_stored_vectors: np.ndarray | None,
    new_stored_vectors: np.ndarray,
    old_lsh: SimHashLSH | None,
    new_lsh: SimHashLSH,
) -> dict[str, float]:
    new_norm = np.linalg.norm(new_cls_reps, axis=1)
    stats: dict[str, float] = {
        "lsh/representation_norm_mean": float(new_norm.mean()),
        "lsh/representation_norm_std": float(new_norm.std()),
    }
    if old_cls_reps is None or old_stored_vectors is None or old_lsh is None:
        return stats
    old_norm = np.linalg.norm(old_cls_reps, axis=1)
    denom = np.maximum(old_norm * new_norm, 1e-12)
    cosines = np.sum(old_cls_reps * new_cls_reps, axis=1) / denom
    drift = 1.0 - cosines
    changed = 0
    total = 0
    for table_id in range(new_lsh.l):
        for index in range(new_stored_vectors.shape[0]):
            old_key = old_lsh.hash_vector(old_stored_vectors[index], table_id)
            new_key = new_lsh.hash_vector(new_stored_vectors[index], table_id)
            changed += int(old_key != new_key)
            total += 1
    stats.update(
        {
            "lsh/representation_cosine_mean": float(np.mean(cosines)),
            "lsh/representation_cosine_drift_mean": float(np.mean(drift)),
            "lsh/representation_cosine_drift_p95": float(np.percentile(drift, 95)),
            "lsh/bucket_assignment_changed_frac": float(changed / max(total, 1)),
        }
    )
    return stats


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
    if args.refresh_lsh == "every_100_steps":
        return step > 0 and step % 100 == 0
    if args.refresh_lsh == "every_50_steps":
        return step > 0 and step % 50 == 0
    if args.refresh_lsh == "every_25_steps":
        return step > 0 and step % 25 == 0
    if args.refresh_lsh == "every_eval":
        return step > 0 and step % args.eval_every == 0
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


def sample_summary_row(step: int, epoch: float, sample, labels: torch.Tensor) -> dict:
    labels_cpu = labels.detach().cpu().numpy().astype("int64")
    probabilities = np.asarray(sample.probabilities, dtype="float64")
    bucket_sizes = np.asarray(sample.bucket_sizes, dtype="float64")
    unique_count = len(set(sample.indices))
    duplicate_count = len(sample.indices) - unique_count
    label0_count = int(np.sum(labels_cpu == 0))
    label1_count = int(np.sum(labels_cpu == 1))
    duplicate_frac = duplicate_count / max(len(sample.indices), 1)
    return {
        "global_step": step,
        "epoch": epoch,
        "sample/unique_count": unique_count,
        "sample/duplicate_count": duplicate_count,
        "sample/duplicate_frac": duplicate_frac,
        "sample/label0_count": label0_count,
        "sample/label1_count": label1_count,
        "sample/fallback_count": int(np.sum(sample.fallback_flags)),
        "sample/mean_p": float(probabilities.mean()) if probabilities.size else 0.0,
        "sample/min_p": float(probabilities.min()) if probabilities.size else 0.0,
        "sample/max_p": float(probabilities.max()) if probabilities.size else 0.0,
        "sample/mean_bucket_size": float(bucket_sizes.mean()) if bucket_sizes.size else 0.0,
        "sample/resample_count_sum": int(np.sum(sample.resample_counts)),
        "sampler/unique_count_batch": unique_count,
        "sampler/duplicate_count_batch": duplicate_count,
        "sampler/duplicate_frac_batch": duplicate_frac,
        "sampler/label0_count_batch": label0_count,
        "sampler/label1_count_batch": label1_count,
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

    random_sampler = RandomBatchSampler(len(data.train), seed=args.seed + 101, mode=args.random_mode)
    random_sampler.reset_epoch()
    if args.variant == "random":
        sampler = random_sampler
        cls_reps = None
        stored_vectors = None
        lsh = None
        refresh_count = 0
        initial_refresh_time = 0.0
    elif args.sampler_start_step > 0:
        sampler = random_sampler
        cls_reps = None
        stored_vectors = None
        lsh = None
        refresh_count = 0
        initial_refresh_time = 0.0
    else:
        cls_reps, stored_vectors, lsh, initial_refresh_time = build_lsh_state(args, model, data.train, labels_np, device)
        sampler = LGDBatchSampler(
            lsh,
            stored_vectors,
            len(data.train),
            seed=args.seed + 101,
            probability_eps=args.probability_eps,
            replacement_mode=args.replacement_mode,
        )
        sampler.reset_epoch()
        refresh_count = 1
        summary = lsh.summary().__dict__
        (run_dir / "initial_lsh_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    start_time = time.time()
    last_eval = None
    status = "finished"
    error_message = None
    epoch_counts: Counter[int] = Counter()
    total_counts: Counter[int] = Counter()
    coverage_rows: list[dict] = []
    sample_log_handle = None
    sample_log_writer = None
    sample_log_fields = [
        "global_step",
        "epoch",
        "sample_position_in_batch",
        "train_index",
        "label",
        "source",
        "fallback",
        "table_id_used",
        "bucket_key",
        "bucket_size",
        "attempts",
        "p_i",
        "cp_i",
        "cpK_i",
        "cos_i",
        "duplicate_within_batch",
        "is_duplicate_within_batch",
        "seen_count_so_far",
        "times_seen_this_epoch_before",
        "resample_count",
    ]
    if args.log_sample_indices:
        sample_log_handle = gzip.open(run_dir / "sample_log.csv.gz", "at", newline="")
        sample_log_writer = csv.DictWriter(sample_log_handle, fieldnames=sample_log_fields)
        sample_log_writer.writeheader()

    try:
        for step in range(1, total_steps + 1):
            step_start = time.time()
            refresh_time = 0.0
            refresh_drift_metrics: dict[str, float] = {}
            use_lgd_sampler = args.variant != "random" and step > args.sampler_start_step
            if use_lgd_sampler and lsh is None:
                cls_reps, stored_vectors, lsh, initial_refresh_time = build_lsh_state(args, model, data.train, labels_np, device)
                sampler = LGDBatchSampler(
                    lsh,
                    stored_vectors,
                    len(data.train),
                    seed=args.seed + 101,
                    probability_eps=args.probability_eps,
                    replacement_mode=args.replacement_mode,
                )
                sampler.reset_epoch()
                refresh_count += 1
                summary = lsh.summary().__dict__
                (run_dir / "initial_lsh_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

            if use_lgd_sampler and should_refresh(args, step - 1, steps_per_epoch):
                old_cls_reps = cls_reps
                old_stored_vectors = stored_vectors
                old_lsh = lsh
                cls_reps, stored_vectors, lsh, refresh_time = build_lsh_state(args, model, data.train, labels_np, device)
                refresh_drift_metrics = representation_drift_stats(
                    old_cls_reps,
                    cls_reps,
                    old_stored_vectors,
                    stored_vectors,
                    old_lsh,
                    lsh,
                )
                sampler.update_index(lsh, stored_vectors)
                refresh_count += 1

            epoch = (step - 1) / max(steps_per_epoch, 1)
            if args.variant == "random" or not use_lgd_sampler:
                sample = sampler.sample_batch(args.batch_size)
                query_norm = 0.0
            else:
                query = classifier_query(model, args.variant, include_bias=args.include_bias_in_lsh)
                query_norm = float(np.linalg.norm(query))
                sample = sampler.sample_batch(args.batch_size, query)

            batch = data.train.batch(sample.indices, device)
            labels = batch.pop("labels")
            step_sample_summary = sample_summary_row(step, epoch, sample, labels)
            if args.log_sample_indices and sample_log_writer is not None:
                batch_counts = Counter(int(index) for index in sample.indices)
                seen_within_batch: Counter[int] = Counter()
                labels_cpu = labels.detach().cpu().numpy().astype("int64")
                for pos, index in enumerate(sample.indices):
                    index = int(index)
                    seen_within_batch[index] += 1
                    duplicate_within_batch = bool(batch_counts[index] > 1)
                    sample_log_writer.writerow(
                        {
                            "global_step": step,
                            "epoch": epoch,
                            "sample_position_in_batch": pos,
                            "train_index": index,
                            "label": int(labels_cpu[pos]),
                            "source": sample.sources[pos],
                            "fallback": bool(sample.fallback_flags[pos]),
                            "table_id_used": sample.table_ids[pos],
                            "bucket_key": sample.bucket_keys[pos],
                            "bucket_size": int(sample.bucket_sizes[pos]),
                            "attempts": int(sample.attempts[pos]),
                            "p_i": float(sample.probabilities[pos]),
                            "cp_i": float(sample.collision_probs[pos]),
                            "cpK_i": float(sample.collision_probs_k[pos]),
                            "cos_i": float(sample.cosines[pos]),
                            "duplicate_within_batch": duplicate_within_batch,
                            "is_duplicate_within_batch": duplicate_within_batch,
                            "seen_count_so_far": int(total_counts[index] + seen_within_batch[index]),
                            "times_seen_this_epoch_before": int(epoch_counts[index]),
                            "resample_count": int(sample.resample_counts[pos]),
                        }
                    )
                sample_log_handle.flush()
                write_csv_row(run_dir / "sample_step_log.csv", step_sample_summary)
            if args.audit_sample_coverage or args.log_sample_indices:
                epoch_counts.update(int(index) for index in sample.indices)
                total_counts.update(int(index) for index in sample.indices)
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
            train_metrics.update(step_sample_summary)
            train_metrics["sampler/coverage_so_far"] = len(total_counts) / max(len(data.train), 1)
            train_metrics.update(refresh_drift_metrics)
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
            if (args.audit_sample_coverage or args.log_sample_indices) and (step % steps_per_epoch == 0 or step == total_steps):
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
                if hasattr(sampler, "reset_epoch"):
                    sampler.reset_epoch()
    except Exception as exc:
        status = "failed"
        error_message = repr(exc)
        raise
    finally:
        if sample_log_handle is not None:
            sample_log_handle.close()
        summary = {
            "status": status,
            "error_message": error_message,
            "run_dir": str(run_dir),
            "wandb_url": (run.url if run is not None else None),
            "final_eval": last_eval,
            "refresh_count": refresh_count,
            "model_state_hash": model_state_hash(model),
            "sample_coverage_by_epoch": coverage_rows,
            "sample_coverage_total": {
                "train_size": len(data.train),
                "draw_count": sum(total_counts.values()),
                "unique_seen_count": len(total_counts),
                "coverage": len(total_counts) / max(len(data.train), 1),
                "duplicate_draw_count": sum(value - 1 for value in total_counts.values() if value > 1),
            },
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
    parser.add_argument(
        "--refresh_lsh",
        choices=["initial_only", "every_epoch", "every_n_steps", "every_100_steps", "every_50_steps", "every_25_steps", "every_eval", "no_refresh"],
        default="every_epoch",
    )
    parser.add_argument("--refresh_steps", type=int, default=100)
    parser.add_argument("--replacement_mode", choices=["with_replacement", "batch_without_replacement", "epoch_no_reuse"], default="with_replacement")
    parser.add_argument("--random_mode", choices=["with_replacement", "epoch_shuffle"], default="with_replacement")
    parser.add_argument("--sampler_start_step", type=int, default=0)
    parser.add_argument("--sampler_warmup", choices=["random", "none", "disabled"], default="none")
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
    parser.add_argument("--log_sample_indices", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = train_main(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

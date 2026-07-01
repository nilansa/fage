#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
FIX_ROOT = CODE_ROOT.parents[1]
RUNS_ROOT = FIX_ROOT / "runs"
REPORTS_ROOT = FIX_ROOT / "reports"
LOGS_ROOT = FIX_ROOT / "logs"
PLOTS_ROOT = FIX_ROOT / "plots"
SOURCE_ROOT = Path("/home2/nilan/research/AN/scratch_exps/fage_sherlock_tmp/bert-lgd-reproduction/lgd_bert_sherlock_audit_20260628")
PAPER_MAIN = Path("/home2/nilan/research/AN/scratch_exps/papers/Experiments-NeurIPS-2019-fast-and-accurate-stochastic-gradient-estimation-Paper.pdf")
PAPER_SUPPLEMENT = Path("/home2/nilan/research/AN/scratch_exps/papers/experiments-lgd_supplement.pdf")
WANDB_PROJECT = "lgd_bert_sampling_improvement_mrpc"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value in {"", None} else float(value)
    except Exception:
        return default


def detect_gpus() -> list[int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,name", "--format=csv,noheader,nounits"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit("No GPUs detected by nvidia-smi.")
    gpus: list[int] = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if parts:
            gpus.append(int(parts[0]))
    return sorted(gpus)


def wandb_mode_from_env(requested: str) -> str:
    if requested != "auto":
        return requested
    if os.environ.get("WANDB_API_KEY"):
        return "online"
    netrc = Path.home() / ".netrc"
    if netrc.exists():
        try:
            if "api.wandb.ai" in netrc.read_text(errors="ignore"):
                return "online"
        except Exception:
            pass
    return "offline"


def jobs() -> list[dict[str, object]]:
    base = {
        "task": "mrpc",
        "seed": 0,
        "batch_size": 32,
        "epochs": 3,
        "lr": 2e-5,
        "eval_every": 25,
        "correction": "none",
        "refresh_lsh": "every_epoch",
        "sampler_start_step": 0,
        "sampler_warmup": "none",
    }
    scheduled: list[dict[str, object]] = [
        {
            **base,
            "run_name": "MRPC_random_epoch_shuffle_seed0",
            "variant": "random",
            "random_mode": "epoch_shuffle",
            "replacement_mode": "with_replacement",
            "lsh_k": 0,
            "lsh_l": 0,
        }
    ]
    for k in [3, 4, 5, 6]:
        scheduled.append(
            {
                **base,
                "run_name": f"MRPC_LA_K{k}_L50_BWoR_refreshEpoch_noCorr_seed0",
                "variant": "label_aware_lgd",
                "random_mode": "epoch_shuffle",
                "replacement_mode": "batch_without_replacement",
                "lsh_k": k,
                "lsh_l": 50,
            }
        )
    return scheduled


def command_for_job(job: dict[str, object], run_dir: Path, wandb_mode: str) -> list[str]:
    cmd = [
        sys.executable,
        str(CODE_ROOT / "scripts" / "run_lgd_bert.py"),
        "--task",
        str(job["task"]),
        "--variant",
        str(job["variant"]),
        "--correction",
        str(job["correction"]),
        "--seed",
        str(job["seed"]),
        "--run_name",
        str(job["run_name"]),
        "--run_dir",
        str(run_dir),
        "--batch_size",
        str(job["batch_size"]),
        "--epochs",
        str(job["epochs"]),
        "--lr",
        str(job["lr"]),
        "--optimizer",
        "adam",
        "--eval_every",
        str(job["eval_every"]),
        "--wandb_project",
        WANDB_PROJECT,
        "--wandb_mode",
        wandb_mode,
        "--wandb_group",
        "sampling_improvement_mrpc",
        "--lsh_k",
        str(job["lsh_k"]),
        "--lsh_l",
        str(job["lsh_l"]),
        "--refresh_lsh",
        str(job["refresh_lsh"]),
        "--replacement_mode",
        str(job["replacement_mode"]),
        "--random_mode",
        str(job["random_mode"]),
        "--sampler_start_step",
        str(job["sampler_start_step"]),
        "--sampler_warmup",
        str(job["sampler_warmup"]),
        "--correction_max_weight",
        "10.0",
        "--audit_sample_coverage",
        "--log_sample_indices",
    ]
    return cmd


def prepare_run_dir(run_name: str) -> Path:
    run_dir = RUNS_ROOT / run_name
    if run_dir.exists():
        archived = RUNS_ROOT / f"{run_name}.prelaunch_archive_{int(time.time())}"
        run_dir.rename(archived)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def summarize_run(job: dict[str, object], manifest: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    run_dir = Path(str(manifest["run_dir"]))
    eval_rows = read_csv_rows(run_dir / "eval_log.csv")
    train_rows = read_csv_rows(run_dir / "train_log.csv")
    final_eval = eval_rows[-1] if eval_rows else {}
    best_acc = max((as_float(row, "eval/accuracy") for row in eval_rows), default=0.0)
    best_f1 = max((as_float(row, "eval/f1") for row in eval_rows), default=0.0)
    final_train = train_rows[-1] if train_rows else {}

    label0 = sum(as_float(row, "sampler/label0_count_batch") for row in train_rows)
    label1 = sum(as_float(row, "sampler/label1_count_batch") for row in train_rows)
    denom = max(label0 + label1, 1.0)

    metric_row = {
        "run_name": job["run_name"],
        "K": "" if int(job["lsh_k"]) == 0 else job["lsh_k"],
        "L": "" if int(job["lsh_l"]) == 0 else job["lsh_l"],
        "replacement_mode": job["random_mode"] if job["variant"] == "random" else job["replacement_mode"],
        "correction": job["correction"],
        "refresh_mode": job["refresh_lsh"],
        "final_eval_accuracy": as_float(final_eval, "eval/accuracy"),
        "final_eval_f1": as_float(final_eval, "eval/f1"),
        "final_eval_loss": as_float(final_eval, "eval/loss"),
        "best_eval_accuracy": best_acc,
        "best_eval_f1": best_f1,
    }
    health_row = {
        "run_name": job["run_name"],
        "K": metric_row["K"],
        "coverage_after_training": as_float(final_train, "sampler/coverage_so_far"),
        "mean_unique_samples_per_batch": sum(as_float(row, "sampler/unique_count_batch") for row in train_rows) / max(len(train_rows), 1),
        "mean_duplicate_samples_per_batch": sum(as_float(row, "sampler/duplicate_count_batch") for row in train_rows) / max(len(train_rows), 1),
        "mean_fallback_rate": sum(as_float(row, "sampler/fallback_rate") for row in train_rows) / max(len(train_rows), 1),
        "mean_bucket_size": sum(as_float(row, "sampler/bucket_size_mean") for row in train_rows) / max(len(train_rows), 1),
        "sampled_label1_ratio": label1 / denom,
    }
    return metric_row, health_row


def markdown_table(rows: list[dict[str, object]], fields: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out) + "\n"


def make_report(metric_rows: list[dict[str, object]], health_rows: list[dict[str, object]], manifests: list[dict[str, object]], wandb_mode: str) -> None:
    final_fields = [
        "run_name",
        "K",
        "L",
        "replacement_mode",
        "correction",
        "refresh_mode",
        "final_eval_accuracy",
        "final_eval_f1",
        "final_eval_loss",
        "best_eval_accuracy",
        "best_eval_f1",
    ]
    health_fields = [
        "run_name",
        "K",
        "coverage_after_training",
        "mean_unique_samples_per_batch",
        "mean_duplicate_samples_per_batch",
        "mean_fallback_rate",
        "mean_bucket_size",
        "sampled_label1_ratio",
    ]
    write_csv_rows(REPORTS_ROOT / "final_metrics.csv", metric_rows, final_fields)
    write_csv_rows(
        REPORTS_ROOT / "sampling_health.csv",
        health_rows,
        health_fields,
    )

    manifest_fields = [
        "run_name",
        "status",
        "gpu",
        "return_code",
        "run_dir",
        "log_path",
        "command",
        "wandb_project",
        "wandb_mode",
        "wandb_url",
        "wandb_offline_dir",
    ]
    write_csv_rows(REPORTS_ROOT / "run_manifest.csv", manifests, manifest_fields)

    random_metrics = next((row for row in metric_rows if row["run_name"] == "MRPC_random_epoch_shuffle_seed0"), None)
    lgd_metrics = [row for row in metric_rows if row["run_name"] != "MRPC_random_epoch_shuffle_seed0"]
    lgd_health = [row for row in health_rows if row["run_name"] != "MRPC_random_epoch_shuffle_seed0"]
    best_health = max(
        lgd_health,
        key=lambda row: (
            float(row["coverage_after_training"]),
            float(row["mean_unique_samples_per_batch"]),
            -float(row["mean_duplicate_samples_per_batch"]),
            -float(row["mean_fallback_rate"]),
        ),
        default=None,
    )
    best_metric = max(
        lgd_metrics,
        key=lambda row: (float(row["best_eval_f1"]), float(row["best_eval_accuracy"])),
        default=None,
    )
    random_beaten = False
    if random_metrics and best_metric:
        random_beaten = (
            float(best_metric["best_eval_f1"]) > float(random_metrics["best_eval_f1"])
            or float(best_metric["best_eval_accuracy"]) > float(random_metrics["best_eval_accuracy"])
        )
    coverage_fixed = all(float(row["coverage_after_training"]) > 0.10 for row in lgd_health)
    duplicates_fixed = all(float(row["mean_duplicate_samples_per_batch"]) < 10.0 for row in lgd_health)
    sampling_improved = all(
        float(row["coverage_after_training"]) > 0.001 and float(row["mean_duplicate_samples_per_batch"]) < 29.0
        for row in lgd_health
    )

    if not sampling_improved:
        verdict = "Sampling is still broken; next fix should be supplement-style mini-batch collection, not probability correction."
    elif coverage_fixed and duplicates_fixed and not random_beaten:
        verdict = "Coverage is no longer the main failure; next check should be query-loss correlation / whether label-aware score actually selects high-loss MRPC examples."
    else:
        best_k = best_metric.get("K") if best_metric else "unknown"
        verdict = f"Use K={best_k} as the next candidate for a refresh-rate sweep."

    paper_note = REPORTS_ROOT / "paper_notes.md"
    paper_note.write_text(
        "\n".join(
            [
                "# Paper Notes",
                "",
                f"- Main PDF: `{PAPER_MAIN}`.",
                "- Main PDF anchors used here: BERTbase on MRPC/RTE; 3 epochs; batch size 32; Adam; LSH K=7, L=10; K controls collision-probability decay; logistic LGD stores label-multiplied inputs and queries with negative classifier direction.",
                f"- Supplement PDF: `{PAPER_SUPPLEMENT}`.",
                "- Supplement PDF anchors used here: BERT pooled representations are stored in LSH tables, tables can be periodically refreshed, and classifier-layer parameters are used as sampling queries.",
                "- The supplement-style mini-batch bucket-filling rule was taken from the run prompt; it was not separately visible in the extracted two-page local supplement text.",
                "",
            ]
        )
    )

    commands = "\n".join(f"- GPU {row['gpu']}: `{row['command']}`" for row in manifests)
    wandb_rows = "\n".join(
        f"- `{row['run_name']}`: {row.get('wandb_url') or row.get('wandb_offline_dir') or 'not recorded'}"
        for row in manifests
    )
    report = f"""# MRPC Sampling Improvement Report

## Folder

Created folder: `{FIX_ROOT}`

Copied source folder: `{SOURCE_ROOT}`

## Paper Files

- `{PAPER_MAIN}` ({'found' if PAPER_MAIN.exists() else 'missing'})
- `{PAPER_SUPPLEMENT}` ({'found' if PAPER_SUPPLEMENT.exists() else 'missing'})

See `paper_notes.md` for the paper anchors used in this run.

## Commands And GPU Assignment

{commands}

## W&B

Project: `{WANDB_PROJECT}`

Mode: `{wandb_mode}`

Runs / offline paths:

{wandb_rows}

## Final MRPC Metrics

{markdown_table(metric_rows, final_fields)}

## Sampling Health

{markdown_table(health_rows, health_fields)}

## Verdict

- Sampling improved versus the collapsed K=7,L=10 reference: `{sampling_improved}`.
- Best K by sampling health: `{best_health.get('K') if best_health else 'n/a'}`.
- Best K by MRPC accuracy/F1: `{best_metric.get('K') if best_metric else 'n/a'}`.
- Any LGD variant beat random epoch-shuffle: `{random_beaten}`.
- Decision: {verdict}
"""
    (REPORTS_ROOT / "MRPC_SAMPLING_IMPROVEMENT_REPORT.md").write_text(report)


def launch(args: argparse.Namespace) -> int:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)

    wandb_mode = wandb_mode_from_env(args.wandb_mode)
    selected_gpus = [int(item) for item in args.gpus.split(",")] if args.gpus else detect_gpus()[:4]
    if not selected_gpus:
        raise SystemExit("No GPUs available.")

    scheduled = jobs()
    manifests: list[dict[str, object]] = []
    pending: list[dict[str, object]] = []
    for job in scheduled:
        run_name = str(job["run_name"])
        run_dir = prepare_run_dir(run_name)
        log_path = LOGS_ROOT / f"{run_name}.log"
        cmd = command_for_job(job, run_dir, wandb_mode)
        (run_dir / "command.txt").write_text(shlex.join(cmd) + "\n")
        manifest = {
            "run_name": run_name,
            "status": "pending",
            "gpu": "",
            "return_code": "",
            "run_dir": str(run_dir),
            "log_path": str(log_path),
            "command": shlex.join(cmd),
            "wandb_project": WANDB_PROJECT,
            "wandb_mode": wandb_mode,
            "wandb_url": "",
            "wandb_offline_dir": str(run_dir / "wandb"),
        }
        manifests.append(manifest)
        pending.append({"job": job, "manifest": manifest, "cmd": cmd, "run_dir": run_dir, "log_path": log_path})

    if args.dry_run:
        make_report([], [], manifests, wandb_mode)
        print(f"DRY_RUN_MANIFEST={REPORTS_ROOT / 'run_manifest.csv'}")
        return 0

    active: list[dict[str, object]] = []
    while pending or active:
        active_gpus = {int(item["gpu"]) for item in active}
        free_gpus = [gpu for gpu in selected_gpus if gpu not in active_gpus]
        while pending and free_gpus:
            gpu = free_gpus.pop(0)
            item = pending.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["WANDB_PROJECT"] = WANDB_PROJECT
            env["WANDB_MODE"] = wandb_mode
            env["WANDB_DIR"] = str(Path(str(item["run_dir"])) / "wandb")
            previous_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(CODE_ROOT / "src") + (os.pathsep + previous_pythonpath if previous_pythonpath else "")
            env.setdefault("HF_HOME", "/ssd_scratch/nilan/.cache/huggingface")
            env.setdefault("HF_DATASETS_CACHE", "/ssd_scratch/nilan/.cache/huggingface/datasets")

            log_handle = Path(str(item["log_path"])).open("w")
            manifest = item["manifest"]
            manifest["status"] = "running"
            manifest["gpu"] = gpu
            print(f"START gpu={gpu} {manifest['run_name']}", flush=True)
            proc = subprocess.Popen(
                item["cmd"],
                cwd=str(CODE_ROOT),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            active.append({**item, "proc": proc, "gpu": gpu, "log_handle": log_handle, "start": time.time()})

        time.sleep(args.poll_sec)
        still_active: list[dict[str, object]] = []
        for item in active:
            proc = item["proc"]
            code = proc.poll()
            if code is None:
                still_active.append(item)
                continue
            item["log_handle"].close()
            manifest = item["manifest"]
            manifest["return_code"] = code
            manifest["status"] = "finished" if code == 0 else "failed"
            wandb_url_path = Path(str(manifest["run_dir"])) / "wandb_url.txt"
            if wandb_url_path.exists():
                manifest["wandb_url"] = wandb_url_path.read_text().strip()
            print(f"FINISH gpu={item['gpu']} {manifest['run_name']} returncode={code}", flush=True)
        active = still_active

    metric_rows: list[dict[str, object]] = []
    health_rows: list[dict[str, object]] = []
    by_name = {str(job["run_name"]): job for job in scheduled}
    for manifest in manifests:
        if manifest["status"] != "finished":
            continue
        metric_row, health_row = summarize_run(by_name[str(manifest["run_name"])], manifest)
        metric_rows.append(metric_row)
        health_rows.append(health_row)

    make_report(metric_rows, health_rows, manifests, wandb_mode)
    print(markdown_table(metric_rows, ["run_name", "K", "final_eval_accuracy", "final_eval_f1", "best_eval_accuracy", "best_eval_f1"]))
    failures = [row for row in manifests if row["status"] != "finished"]
    if failures:
        print(f"FAILED_RUNS={','.join(str(row['run_name']) for row in failures)}", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch MRPC-only LGD sampling-improvement full runs.")
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU ids. Defaults to first four detected GPUs.")
    parser.add_argument("--wandb_mode", choices=["auto", "online", "offline", "disabled"], default="auto")
    parser.add_argument("--poll_sec", type=float, default=5.0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    raise SystemExit(launch(args))


if __name__ == "__main__":
    main()

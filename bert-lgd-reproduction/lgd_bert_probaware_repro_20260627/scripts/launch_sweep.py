#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def detect_gpus() -> list[int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,memory.total,name", "--format=csv,noheader,nounits"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return []
    rows = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        rows.append((int(parts[0]), int(parts[1]), int(parts[2]), parts[3]))
    rows.sort(key=lambda row: row[1], reverse=True)
    return [row[0] for row in rows]


def jobs_for_stage(stage: str, seeds: list[int]) -> list[dict]:
    if stage == "smoke":
        return [
            {"task": "mrpc", "variant": "random", "correction": "none", "seed": 0, "max_steps": 20},
            {"task": "mrpc", "variant": "label_aware_lgd", "correction": "none", "seed": 0, "max_steps": 20},
            {"task": "mrpc", "variant": "label_aware_lgd", "correction": "full", "seed": 0, "max_steps": 20},
            {"task": "rte", "variant": "random", "correction": "none", "seed": 0, "max_steps": 20},
            {"task": "rte", "variant": "label_aware_lgd", "correction": "none", "seed": 0, "max_steps": 20},
            {"task": "rte", "variant": "label_aware_lgd", "correction": "full", "seed": 0, "max_steps": 20},
        ]
    if stage == "sanity":
        return [
            {"task": task, "variant": variant, "correction": corr, "seed": 0, "max_steps": 100}
            for task in ["mrpc", "rte"]
            for variant, corr in [("random", "none"), ("paper_lgd", "none"), ("label_aware_lgd", "full")]
        ]
    if stage == "full_pair":
        return [
            {"task": "mrpc", "variant": "random", "correction": "none", "seed": 0, "max_steps": None},
            {"task": "mrpc", "variant": "label_aware_lgd", "correction": "full", "seed": 0, "max_steps": None},
        ]
    if stage == "coverage_audit":
        return [
            {"task": task, "variant": variant, "correction": "none", "seed": 0, "max_steps": None}
            for task in ["mrpc", "rte"]
            for variant in ["random", "paper_lgd", "label_aware_lgd"]
        ]
    if stage == "full_sweep":
        jobs = []
        for task in ["mrpc", "rte"]:
            for seed in seeds:
                jobs.append({"task": task, "variant": "random", "correction": "none", "seed": seed, "max_steps": None})
                for variant in ["paper_lgd", "label_aware_lgd"]:
                    for correction in ["none", "full"]:
                        jobs.append({"task": task, "variant": variant, "correction": correction, "seed": seed, "max_steps": None})
        return jobs
    raise ValueError(f"Unsupported stage: {stage}")


def command_for_job(job: dict, args) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_lgd_bert.py"),
        "--task",
        job["task"],
        "--variant",
        job["variant"],
        "--correction",
        job["correction"],
        "--seed",
        str(job["seed"]),
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--eval_every",
        str(args.eval_every),
        "--wandb_project",
        args.wandb_project,
        "--wandb_mode",
        args.wandb_mode,
        "--wandb_group",
        args.wandb_group,
        "--output_root",
        str(args.run_root),
        "--refresh_lsh",
        args.refresh_lsh,
    ]
    if job["max_steps"] is not None:
        cmd += ["--max_steps", str(job["max_steps"])]
    if args.correction_max_weight is not None:
        cmd += ["--correction_max_weight", str(args.correction_max_weight)]
    if args.audit_sample_coverage:
        cmd += ["--audit_sample_coverage"]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch queued independent BERT LGD runs across visible GPUs.")
    parser.add_argument("--stage", choices=["smoke", "sanity", "full_pair", "full_sweep", "coverage_audit"], default="smoke")
    parser.add_argument("--allow_full", action="store_true", help="Required for full_pair and full_sweep.")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--eval_every", type=int, default=25)
    parser.add_argument("--wandb_project", default="lgd_neurips2019_bert_repro")
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--refresh_lsh", choices=["initial_only", "every_epoch", "every_n_steps", "no_refresh"], default="every_epoch")
    parser.add_argument("--correction_max_weight", type=float, default=10.0)
    parser.add_argument("--audit_sample_coverage", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.stage.startswith("full") and not args.allow_full:
        raise SystemExit("Refusing full runs without --allow_full. Run smoke/sanity gates first.")
    if args.wandb_group is None:
        args.wandb_group = f"{args.stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.run_root = ROOT / "runs" / args.wandb_group
    args.run_root.mkdir(parents=True, exist_ok=True)

    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    jobs = jobs_for_stage(args.stage, seeds)
    gpus = detect_gpus()
    if not gpus:
        raise SystemExit("No GPUs detected by nvidia-smi.")
    command_log = args.run_root / "commands.txt"
    active: list[tuple[subprocess.Popen, int, object, object]] = []
    pending = list(jobs)
    with command_log.open("w") as log:
        for job in jobs:
            log.write(shlex.join(command_for_job(job, args)) + "\n")
    if args.dry_run:
        print(command_log)
        return

    while pending or active:
        active_gpus = {gpu for _proc, gpu, _stdout, _stderr in active}
        free_gpus = [gpu for gpu in gpus if gpu not in active_gpus]
        while pending and free_gpus:
            gpu = free_gpus.pop(0)
            job = pending.pop(0)
            name = f"{job['task']}_{job['variant']}_{job['correction']}_seed{job['seed']}_{int(time.time())}"
            stdout = (args.run_root / f"{name}.stdout.log").open("w")
            stderr = (args.run_root / f"{name}.stderr.log").open("w")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env.setdefault("HF_HOME", "/ssd_scratch/nilan/.cache/huggingface")
            env.setdefault("HF_DATASETS_CACHE", "/ssd_scratch/nilan/.cache/huggingface/datasets")
            env.setdefault("WANDB_DIR", str(args.run_root / "wandb"))
            cmd = command_for_job(job, args)
            print(f"START gpu={gpu} {shlex.join(cmd)}", flush=True)
            proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=stdout, stderr=stderr)
            active.append((proc, gpu, stdout, stderr))
        time.sleep(5)
        still_active = []
        for proc, gpu, stdout, stderr in active:
            code = proc.poll()
            if code is None:
                still_active.append((proc, gpu, stdout, stderr))
            else:
                stdout.close()
                stderr.close()
                print(f"FINISH gpu={gpu} returncode={code}", flush=True)
                if code != 0:
                    raise SystemExit(f"A run failed with return code {code}; see logs in {args.run_root}")
        active = still_active


if __name__ == "__main__":
    main()

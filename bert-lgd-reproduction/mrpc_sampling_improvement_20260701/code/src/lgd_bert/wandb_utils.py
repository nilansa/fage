from __future__ import annotations

import os
from pathlib import Path


def init_wandb(config: dict, run_dir: Path):
    import wandb

    mode = config.get("wandb_mode", "online")
    if mode == "disabled":
        return None
    os.environ.setdefault("WANDB_PROJECT", config.get("wandb_project", "lgd_bert_sherlock_audit"))
    os.environ.setdefault("WANDB_DIR", str(run_dir / "wandb"))
    run = wandb.init(
        project=config.get("wandb_project", "lgd_bert_sherlock_audit"),
        group=config.get("wandb_group") or None,
        name=config.get("run_name") or None,
        mode=mode,
        dir=os.environ["WANDB_DIR"],
        config=config,
        tags=[
            config.get("task", "unknown"),
            config.get("variant", "unknown"),
            config.get("correction", "unknown"),
        ],
    )
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")
    if run.url:
        (run_dir / "wandb_url.txt").write_text(run.url + "\n")
        print(f"WANDB_URL={run.url}", flush=True)
    return run


def wandb_log(run, metrics: dict) -> None:
    if run is not None:
        run.log(metrics)


def wandb_finish(run) -> None:
    if run is not None:
        run.finish()

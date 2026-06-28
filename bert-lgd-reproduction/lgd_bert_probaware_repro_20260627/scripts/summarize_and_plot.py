#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def float_or_none(value):
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def label_for(config: dict) -> str:
    variant = config["variant"]
    correction = config["correction"]
    if variant == "random":
        return "SGD/random"
    if variant == "paper_lgd" and correction == "none":
        return "paper LGD"
    if variant == "paper_lgd":
        return f"paper LGD {correction}"
    if variant == "label_aware_lgd" and correction == "none":
        return "label-aware LGD"
    if variant == "label_aware_lgd" and correction == "full":
        return "label-aware LGD corrected"
    return f"{variant} {correction}"


def gather(run_root: Path) -> tuple[list[dict], list[dict]]:
    finals: list[dict] = []
    curves: list[dict] = []
    for config_path in sorted(run_root.glob("*/config.json")):
        run_dir = config_path.parent
        config = json.loads(config_path.read_text())
        summary_path = run_dir / "run_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        eval_rows = read_csv(run_dir / "eval_log.csv")
        train_rows = read_csv(run_dir / "train_log.csv")
        last_train = train_rows[-1] if train_rows else {}
        final_eval = summary.get("final_eval") or (eval_rows[-1] if eval_rows else {})
        base = {
            "run_dir": str(run_dir),
            "task": config.get("task"),
            "variant": config.get("variant"),
            "correction": config.get("correction"),
            "label": label_for(config),
            "seed": config.get("seed"),
            "wandb_url": summary.get("wandb_url"),
            "status": summary.get("status"),
            "refresh_count": summary.get("refresh_count"),
            "final_step": final_eval.get("global_step"),
            "final_accuracy": final_eval.get("eval/accuracy") or final_eval.get("paper_fig5/test_accuracy"),
            "final_loss": final_eval.get("eval/loss") or final_eval.get("paper_fig5/test_loss"),
            "final_f1": final_eval.get("eval/f1"),
            "wall_clock_sec": final_eval.get("time/wall_clock_sec"),
            "fallback_rate_last": last_train.get("sampler/fallback_rate"),
            "p_mean_last": last_train.get("sampler/p_mean"),
            "weight_mean_last": last_train.get("correction/weight_mean"),
            "weight_max_last": last_train.get("correction/weight_max"),
            "weight_clipped_frac_last": last_train.get("correction/weight_clipped_frac"),
        }
        finals.append(base)
        for row in eval_rows:
            curves.append(
                {
                    "task": config.get("task"),
                    "label": label_for(config),
                    "seed": config.get("seed"),
                    "global_step": int(float(row["global_step"])),
                    "accuracy": float(row["paper_fig5/test_accuracy"]),
                    "loss": float(row["paper_fig5/test_loss"]),
                }
            )
    return finals, curves


def write_table(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def mean_curves(curves: list[dict]) -> dict:
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in curves:
        grouped[(row["task"], row["label"], row["global_step"])].append(row)
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for (task, label, step), rows in grouped.items():
        out[(task, label)].append(
            {
                "global_step": step,
                "accuracy": sum(row["accuracy"] for row in rows) / len(rows),
                "loss": sum(row["loss"] for row in rows) / len(rows),
                "n": len(rows),
            }
        )
    for rows in out.values():
        rows.sort(key=lambda row: row["global_step"])
    return out


def plot(curves: list[dict], out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    averaged = mean_curves(curves)
    colors = {
        "SGD/random": "#1f77b4",
        "paper LGD": "#d62728",
        "paper LGD full": "#9467bd",
        "label-aware LGD": "#2ca02c",
        "label-aware LGD corrected": "#ff7f0e",
    }
    paths: list[Path] = []
    for task in ["mrpc", "rte"]:
        for metric, title_metric in [("accuracy", "Testing Accuracy"), ("loss", "Testing Loss")]:
            fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
            for (curve_task, label), rows in sorted(averaged.items()):
                if curve_task != task:
                    continue
                x = [row["global_step"] for row in rows]
                y = [row[metric] for row in rows]
                ax.plot(x, y, marker="o", linewidth=1.8, markersize=3, label=label, color=colors.get(label))
            ax.set_title(f"{task.upper()} {title_metric}")
            ax.set_xlabel("Iter")
            ax.set_ylabel("Accuracy" if metric == "accuracy" else "Loss")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            path = out_dir / f"{task}_{metric}_vs_iter.png"
            fig.savefig(path)
            plt.close(fig)
            paths.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), dpi=160)
    panel_specs = [
        ("mrpc", "accuracy", "MRPC Testing Accuracy"),
        ("rte", "accuracy", "RTE Testing Accuracy"),
        ("mrpc", "loss", "MRPC Testing Loss"),
        ("rte", "loss", "RTE Testing Loss"),
    ]
    for ax, (task, metric, title) in zip(axes.flat, panel_specs):
        for (curve_task, label), rows in sorted(averaged.items()):
            if curve_task != task:
                continue
            ax.plot(
                [row["global_step"] for row in rows],
                [row[metric] for row in rows],
                marker="o",
                linewidth=1.6,
                markersize=2.5,
                label=label,
                color=colors.get(label),
            )
        ax.set_title(title)
        ax.set_xlabel("Iter")
        ax.set_ylabel("Accuracy" if metric == "accuracy" else "Loss")
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    combined = out_dir / "figure5_bert_mrpc_rte_accuracy_loss.png"
    fig.savefig(combined)
    plt.close(fig)
    paths.append(combined)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize BERT LGD runs and export Figure-5-style plots.")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir) if args.out_dir else Path("plots") / run_root.name
    finals, curves = gather(run_root)
    write_table(finals, out_dir / "final_metrics.csv")
    write_table(curves, out_dir / "eval_curves.csv")
    paths = plot(curves, out_dir) if curves else []
    print(out_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()


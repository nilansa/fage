#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> dict:
    try:
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        return {"cmd": cmd, "returncode": result.returncode, "output": result.stdout}
    except FileNotFoundError as exc:
        return {"cmd": cmd, "returncode": 127, "output": repr(exc)}


def module_version(name: str) -> str | None:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception as exc:
        return "unavailable: " + repr(exc)


def main() -> None:
    out_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": stamp,
        "hostname": platform.node(),
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "versions": {
            "torch": module_version("torch"),
            "transformers": module_version("transformers"),
            "datasets": module_version("datasets"),
            "wandb": module_version("wandb"),
            "numpy": module_version("numpy"),
        },
        "commands": {
            "nvidia_smi": run(["nvidia-smi"]),
            "nvidia_smi_query": run(["nvidia-smi", "--query-gpu=index,name,memory.free,memory.total", "--format=csv,noheader"]),
            "free_h": run(["free", "-h"]),
            "df_h": run(["df", "-h", str(ROOT), "/ssd_scratch/nilan"]),
        },
        "disk_usage_root": shutil.disk_usage(ROOT)._asdict(),
    }
    try:
        import torch

        report["torch_cuda"] = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "cuda_version": torch.version.cuda,
        }
    except Exception as exc:
        report["torch_cuda"] = {"error": repr(exc)}
    json_path = out_dir / f"env_report_{stamp}.json"
    txt_path = out_dir / f"env_report_{stamp}.txt"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        f"timestamp: {stamp}",
        f"hostname: {report['hostname']}",
        f"python: {report['python']}",
        f"python_version: {report['python_version']}",
        f"cpu_count: {report['cpu_count']}",
        "",
        "versions:",
    ]
    lines.extend(f"  {k}: {v}" for k, v in report["versions"].items())
    lines.append("")
    for key, value in report["commands"].items():
        lines.append(f"## {key}")
        lines.append(value["output"])
    txt_path.write_text("\n".join(lines) + "\n")
    print(txt_path)
    print(json_path)


if __name__ == "__main__":
    main()


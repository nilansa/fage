from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, load_dataset


TASK_TO_KEYS = {
    "mrpc": ("sentence1", "sentence2"),
    "rte": ("sentence1", "sentence2"),
}


@dataclass
class EncodedDataset:
    task: str
    split: str
    features: dict[str, torch.Tensor]
    labels: torch.Tensor
    raw_size: int

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def batch(self, indices: list[int] | torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
        if not torch.is_tensor(indices):
            indices = torch.tensor(indices, dtype=torch.long)
        batch = {key: value[indices].to(device) for key, value in self.features.items()}
        batch["labels"] = self.labels[indices].to(device)
        return batch


@dataclass
class GlueData:
    train: EncodedDataset
    validation: EncodedDataset
    report: dict


def _encode_split(raw_split, tokenizer, task: str, split: str, max_length: int) -> EncodedDataset:
    key1, key2 = TASK_TO_KEYS[task]
    encoded = tokenizer(
        list(raw_split[key1]),
        list(raw_split[key2]),
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    keep_keys = ["input_ids", "attention_mask", "token_type_ids"]
    features = {
        key: torch.tensor(encoded[key], dtype=torch.long)
        for key in keep_keys
        if key in encoded
    }
    labels = torch.tensor(raw_split["label"], dtype=torch.long)
    return EncodedDataset(task=task, split=split, features=features, labels=labels, raw_size=len(raw_split))


def _load_cached_nyu_glue(task: str):
    cache_root = Path(os.environ.get("HF_DATASETS_CACHE", "/ssd_scratch/nilan/.cache/huggingface/datasets"))
    task_root = cache_root / "nyu-mll___glue" / task / "0.0.0"
    if not task_root.exists():
        return None
    candidates = sorted(path for path in task_root.iterdir() if path.is_dir())
    if not candidates:
        return None
    cache_dir = candidates[-1]
    split_paths = {
        "train": cache_dir / "glue-train.arrow",
        "validation": cache_dir / "glue-validation.arrow",
        "test": cache_dir / "glue-test.arrow",
    }
    if not all(path.exists() for path in split_paths.values()):
        return None
    return DatasetDict({split: Dataset.from_file(str(path)) for split, path in split_paths.items()})


def load_glue_data(task: str, tokenizer, max_length: int = 128) -> GlueData:
    task = task.lower()
    if task not in TASK_TO_KEYS:
        raise ValueError(f"Unsupported GLUE task: {task}")
    raw = _load_cached_nyu_glue(task)
    if raw is not None:
        source = "Local HF Arrow cache: nyu-mll/glue/" + task
    else:
        try:
            raw = load_dataset("glue", task)
            source = "Hugging Face datasets: glue/" + task
        except Exception:
            raw = load_dataset("nyu-mll/glue", task)
            source = "Hugging Face datasets: nyu-mll/glue/" + task
    train = _encode_split(raw["train"], tokenizer, task, "train", max_length)
    validation = _encode_split(raw["validation"], tokenizer, task, "validation", max_length)
    train_counts = torch.bincount(train.labels, minlength=2).tolist()
    val_counts = torch.bincount(validation.labels, minlength=2).tolist()
    paper_expected = {
        "mrpc": {"train": 3669, "validation_or_test": 409},
        "rte": {"train": 2491, "validation_or_test": 278},
    }[task]
    report = {
        "task": task,
        "source": source,
        "train_size": len(train),
        "validation_size": len(validation),
        "train_label_counts": train_counts,
        "validation_label_counts": val_counts,
        "paper_expected_approx": paper_expected,
        "testing_curve_split": "validation",
        "note": "GLUE test labels are unavailable through the standard dataset, so validation is logged as paper_fig5/test_*.",
    }
    return GlueData(train=train, validation=validation, report=report)


def save_data_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import EncodedDataset


def binary_f1(preds: list[int], labels: list[int]) -> float:
    tp = sum(1 for pred, label in zip(preds, labels) if pred == 1 and label == 1)
    fp = sum(1 for pred, label in zip(preds, labels) if pred == 1 and label == 0)
    fn = sum(1 for pred, label in zip(preds, labels) if pred == 0 and label == 1)
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else float(2 * tp / denom)


@torch.no_grad()
def evaluate(model, dataset: EncodedDataset, batch_size: int, device: torch.device) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_count = 0
    preds: list[int] = []
    labels_all: list[int] = []
    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        batch = dataset.batch(indices, device)
        labels = batch.pop("labels")
        outputs = model(**batch)
        logits = outputs.logits
        losses = F.cross_entropy(logits, labels, reduction="none")
        total_loss += float(losses.detach().sum().cpu())
        total_count += int(labels.shape[0])
        preds.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
        labels_all.extend(labels.detach().cpu().tolist())
    if was_training:
        model.train()
    correct = sum(1 for pred, label in zip(preds, labels_all) if pred == label)
    return {
        "eval/loss": total_loss / max(total_count, 1),
        "eval/accuracy": correct / max(total_count, 1),
        "eval/f1": binary_f1(preds, labels_all),
    }

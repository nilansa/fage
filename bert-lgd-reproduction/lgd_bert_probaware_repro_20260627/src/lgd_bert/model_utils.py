from __future__ import annotations

import hashlib

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import EncodedDataset


def load_tokenizer_and_model(model_name: str, num_labels: int = 2):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model


def _base_model(model):
    if hasattr(model, "bert"):
        return model.bert
    prefix = getattr(model, "base_model_prefix", None)
    if prefix and hasattr(model, prefix):
        return getattr(model, prefix)
    if hasattr(model, "base_model"):
        return model.base_model
    raise AttributeError("Could not find the base transformer model for representation extraction")


@torch.no_grad()
def pooled_cls_output(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    base = _base_model(model)
    inputs = {key: value for key, value in batch.items() if key != "labels"}
    outputs = base(**inputs, return_dict=True)
    return outputs.last_hidden_state[:, 0, :]


@torch.no_grad()
def compute_cls_representations(
    model,
    dataset: EncodedDataset,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    was_training = model.training
    model.eval()
    chunks: list[np.ndarray] = []
    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        batch = dataset.batch(indices, device)
        reps = pooled_cls_output(model, batch)
        chunks.append(reps.detach().float().cpu().numpy())
    if was_training:
        model.train()
    return np.concatenate(chunks, axis=0).astype("float32")


def classifier_head(model) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not hasattr(model, "classifier"):
        raise AttributeError("Expected BERT sequence classifier to expose model.classifier")
    weight = model.classifier.weight
    bias = model.classifier.bias
    if weight.shape[0] != 2:
        raise ValueError(f"Expected binary classifier weight shape [2, hidden], got {tuple(weight.shape)}")
    return weight, bias


def classifier_query(model, variant: str, include_bias: bool = False) -> np.ndarray:
    weight, bias = classifier_head(model)
    w0 = weight[0].detach().float().cpu().numpy()
    w1 = weight[1].detach().float().cpu().numpy()
    theta = w1 - w0
    if variant == "paper_lgd":
        query = theta
        bias_term = float((bias[1] - bias[0]).detach().float().cpu()) if bias is not None else 0.0
    elif variant == "label_aware_lgd":
        query = -theta
        bias_term = float((bias[0] - bias[1]).detach().float().cpu()) if bias is not None else 0.0
    else:
        raise ValueError(f"No classifier query for variant: {variant}")
    if include_bias:
        query = np.concatenate([query.astype("float32"), np.array([bias_term], dtype="float32")])
    return query.astype("float32")


def build_stored_vectors(
    cls_reps: np.ndarray,
    labels: np.ndarray,
    variant: str,
    include_bias: bool = False,
) -> np.ndarray:
    vectors = cls_reps.astype("float32", copy=False)
    if include_bias:
        vectors = np.concatenate([vectors, np.ones((vectors.shape[0], 1), dtype="float32")], axis=1)
    if variant == "paper_lgd":
        return vectors.astype("float32", copy=False)
    if variant == "label_aware_lgd":
        ytilde = (labels.astype("int64") * 2 - 1).astype("float32")
        return (vectors * ytilde[:, None]).astype("float32")
    raise ValueError(f"No stored vectors for variant: {variant}")


def model_state_hash(model) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


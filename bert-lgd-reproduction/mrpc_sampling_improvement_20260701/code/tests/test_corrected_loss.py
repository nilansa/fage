from __future__ import annotations

import torch
import torch.nn.functional as F

from lgd_bert.probability import corrected_loss_weights


def test_uniform_probability_correction_equals_mean_ce():
    logits = torch.tensor([[2.0, 0.0], [0.5, 1.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1, 1])
    losses = F.cross_entropy(logits, labels, reduction="none")
    probabilities = torch.full((3,), 1.0 / 99.0)
    weights = corrected_loss_weights(probabilities, train_size=99, mode="full").weights
    assert torch.allclose(weights, torch.ones_like(weights))
    assert torch.allclose((weights * losses).mean(), losses.mean())


def test_classifier_shape_assumption():
    layer = torch.nn.Linear(768, 2)
    assert list(layer.weight.shape) == [2, 768]

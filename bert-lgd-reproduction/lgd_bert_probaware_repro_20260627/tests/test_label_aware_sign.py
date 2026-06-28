from __future__ import annotations

import numpy as np


def test_label_aware_query_scores_wrong_low_margin_examples_higher():
    theta = np.array([1.0, 0.0], dtype="float32")
    h_correct_pos = np.array([2.0, 0.0], dtype="float32")
    h_wrong_pos = np.array([-2.0, 0.0], dtype="float32")
    h_correct_neg = np.array([-2.0, 0.0], dtype="float32")
    h_wrong_neg = np.array([2.0, 0.0], dtype="float32")
    query = -theta
    score_correct_pos = np.dot(+1.0 * h_correct_pos, query)
    score_wrong_pos = np.dot(+1.0 * h_wrong_pos, query)
    score_correct_neg = np.dot(-1.0 * h_correct_neg, query)
    score_wrong_neg = np.dot(-1.0 * h_wrong_neg, query)
    assert score_wrong_pos > score_correct_pos
    assert score_wrong_neg > score_correct_neg


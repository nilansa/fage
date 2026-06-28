# BERT LGD Probability-Aware Reproduction Snapshot

This folder is intended as a review snapshot for the BERT/MRPC/RTE LGD reproduction work done in Codex.

Local source folder on the ADA machine:

```text
/home2/nilan/research/AN/scratch_exps/lgd_bert_probaware_repro_20260627
```

## What Was Built

A self-contained reproduction harness for the NeurIPS 2019 LGD paper's BERT experiment, including:

- random/SGD baseline
- paper-style BERT LGD sampler
- label-aware BERT LGD sampler
- exact probability correction using `1 / (N * p_i)`
- optional clipped/sqrt/normalized correction modes for stability ablations
- W&B logging
- unit tests
- environment inspection
- smoke, sanity, full-pair, and full-sweep launchers
- final metric aggregation and Figure-5-style plot generation

## Important Local Files

Read these first in the local folder:

```text
README.md
reports/FINAL_REPORT.md
reports/STATIC_ANALYSIS_LSH_SAMPLING.md
docs/paper_hyperparams.md
docs/assumptions.md
src/lgd_bert/train.py
src/lgd_bert/samplers.py
src/lgd_bert/probability.py
plots/full_sweep_20260627_1504/final_metrics.csv
plots/full_sweep_20260627_1504/eval_curves.csv
```

## Main Assumptions

- The paper says BERTbase but does not specify cased/uncased; the implementation uses `bert-base-uncased`.
- The rendered PDF shows the learning rate as ambiguous `2e.`; the implementation uses `2e-5`.
- The supplement says to use the final hidden state for the first token; the implementation uses `last_hidden_state[:, 0, :]`.
- GLUE validation is logged as `paper_fig5/test_*` because labeled GLUE test labels are unavailable.
- LSH refresh defaults to every epoch because the supplement says periodic refresh but does not specify the period.

## Sampling Summary

For LGD, each BERT optimizer step uses batch size 32 by drawing 32 independent samples from the sampler. Each selected example keeps its own `p_i`; full correction uses per-example weights:

```python
losses = F.cross_entropy(logits, labels, reduction="none")
weights = 1.0 / (N * p_i)
loss = (weights * losses).mean()
```

Duplicates are allowed because the batch is formed by repeated independent `sample_one()` calls. The completed full-run logs only contain aggregate sampler statistics, not per-step sampled indices.

## Full-Sweep Results

Mean validation accuracy over seeds `0,1,2`:

| Task | Variant | Mean accuracy |
|---|---:|---:|
| MRPC | SGD/random | 0.8309 |
| MRPC | paper LGD | 0.6797 |
| MRPC | paper LGD full | 0.7239 |
| MRPC | label-aware LGD | 0.5743 |
| MRPC | label-aware LGD corrected | 0.6904 |
| RTE | SGD/random | 0.6534 |
| RTE | paper LGD | 0.5800 |
| RTE | paper LGD full | 0.5788 |
| RTE | label-aware LGD | 0.5439 |
| RTE | label-aware LGD corrected | 0.5066 |

In these runs, the random baseline was stronger than the LGD variants. That mismatch from the paper is the main issue to inspect.

## Upload Status

The local folder contains the complete code, docs, reports, metrics, and generated logs. A bulk upload of the full directory needs local GitHub CLI auth (`gh auth login`) or a token-backed `git push`; the current GitHub connector can create text files but cannot stream a local directory path into the repo.

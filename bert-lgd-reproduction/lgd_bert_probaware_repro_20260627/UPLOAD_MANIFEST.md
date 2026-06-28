# Upload Manifest

Snapshot folder:

```text
bert-lgd-reproduction/lgd_bert_probaware_repro_20260627
```

Local complete experiment folder:

```text
/home2/nilan/research/AN/scratch_exps/lgd_bert_probaware_repro_20260627
```

## Uploaded To GitHub

- `README.md`: full approach, assumptions, run summary, limitations, and local file map.
- `final_metrics.csv`: full 30-run sweep final metrics and W&B links.
- `UPLOAD_MANIFEST.md`: this file.

## Present Locally But Not Bulk-Uploaded Yet

The full local folder contains:

- `src/lgd_bert/`: implementation modules.
- `scripts/`: run, launch, inspect, cache, summarize, and plot scripts.
- `tests/`: unit tests for LSH, probability formula, label-aware sign, sampler shapes, and corrected loss.
- `docs/`: paper hyperparameters, assumptions, and reproduction notes.
- `reports/`: final report, static LSH analysis, environment report.
- `plots/full_sweep_20260627_1504/`: metrics CSVs and Figure-5-style PNGs.
- `runs/`: smoke, sanity, full-pair, full-sweep logs and summaries.
- `review_bundle.md`: generated local concatenation of README/docs/reports/source/scripts/tests.

## Why The Whole Directory Is Not Fully Mirrored Yet

The GitHub connector available in this session can create explicit text files in the repository, but it does not expose a local-directory upload or local-path streaming primitive. The local `gh` CLI is also not authenticated, and no GitHub token is available in the shell environment.

For a true full mirror, run `gh auth login` locally or provide token-backed git credentials, then push the local snapshot directory directly.

## Suggested Bulk Push After GitHub CLI Auth

From a temporary checkout of `nilansa/fage`, copy:

```text
/home2/nilan/research/AN/scratch_exps/lgd_bert_probaware_repro_20260627
```

into:

```text
bert-lgd-reproduction/lgd_bert_probaware_repro_20260627
```

Recommended exclusions for GitHub:

```text
**/__pycache__/**
**/*.pyc
**/wandb/**
*.png can be included if desired, but metrics CSVs are already enough for text review.
```

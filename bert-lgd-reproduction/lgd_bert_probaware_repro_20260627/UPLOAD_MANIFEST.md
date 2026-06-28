# Upload Manifest

Snapshot folder:

```text
bert-lgd-reproduction/lgd_bert_probaware_repro_20260627
```

Source folder on ADA:

```text
/home2/nilan/research/AN/scratch_exps/lgd_bert_probaware_repro_20260627
```

## Uploaded

This GitHub folder now contains the full experiment snapshot, including:

- `src/lgd_bert/`: implementation modules.
- `scripts/`: run, launch, inspect, cache, summarize, plot, and sample-coverage scripts.
- `tests/`: unit tests for LSH, probability formula, label-aware sign, sampler shapes, and corrected loss.
- `docs/`: paper hyperparameters, assumptions, and reproduction notes.
- `reports/`: final report, static LSH analysis, environment report.
- `plots/full_sweep_20260627_1504/`: final metrics, eval curves, and Figure-5-style PNGs.
- `runs/`: quick probe, smoke, sanity, full-pair, full-sweep, and sampler-coverage logs/summaries.
- `review_bundle.md`: generated concatenation of README/docs/reports/source/scripts/tests for quick text review.
- `final_metrics.csv`: convenience copy of `plots/full_sweep_20260627_1504/final_metrics.csv`.

## Excluded

Only generated Python bytecode/cache artifacts were excluded:

```text
**/__pycache__/**
**/*.pyc
**/wandb/**
```

The full uploaded folder is about 5.4 MB and contains 458 files.


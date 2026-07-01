# MRPC Sampling Improvement Variant

This folder is a focused evidence bundle for the BERT-LGD MRPC sampling-collapse repair run from
2026-07-01. It is intended for follow-up analysis in ChatGPT Pro or another reviewer, with the
assumptions, commands, code changes, metrics, and sample-index evidence kept together.

## Scope

- Task: MRPC only.
- Seed: 0 only.
- Training: full 3 epochs, batch size 32, Adam, learning rate 2e-5.
- Random baseline: epoch-shuffle random sampling.
- LGD variants: label-aware LGD, `L=50`, `K in {3,4,5,6}`.
- LGD replacement mode: batch without replacement.
- Correction: none.
- LSH refresh: every epoch.
- Warmup: none, LGD starts immediately.

The run intentionally did not include RTE, probability correction, smoke training, sanity training, or
large Cartesian sweeps.

## Headline Result

The immediate duplicate-collapse issue was fixed. All LGD variants produced 32 unique samples per
batch on average, with zero duplicate samples per batch and zero fallback rate. Coverage improved
strongly over the prior collapsed `K=7,L=10` reference, but no LGD variant beat the random
epoch-shuffle baseline on MRPC accuracy/F1.

| run | final acc | final F1 | best acc | best F1 | coverage | dup/batch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random epoch shuffle | 0.8456 | 0.8923 | 0.8627 | 0.9028 | 1.0000 | 0.0 |
| K=3 L=50 BWoR | 0.8260 | 0.8734 | 0.8284 | 0.8829 | 0.9149 | 0.0 |
| K=4 L=50 BWoR | 0.7941 | 0.8547 | 0.7941 | 0.8562 | 0.8528 | 0.0 |
| K=5 L=50 BWoR | 0.7819 | 0.8396 | 0.7819 | 0.8539 | 0.7034 | 0.0 |
| K=6 L=50 BWoR | 0.8211 | 0.8809 | 0.8211 | 0.8809 | 0.5662 | 0.0 |

Best K by sampling health: `K=3`.

Best K by MRPC accuracy/F1 among LGD: `K=3`.

Recommended next action: check query-loss correlation on MRPC. Since coverage and duplicate rate are
no longer the main failure, inspect whether the label-aware score actually selects high-loss MRPC
examples.

## Folder Layout

- `docs/`: inherited reproduction assumptions plus focused scope/pipeline/reviewer notes.
- `reports/`: final report, final metrics, run manifest, sampling health, paper notes, environment.
- `runs/`: per-run command/config/eval/train/sample logs, including `sample_log.csv.gz`.
- `logs/`: combined stdout/stderr logs from each launched run.
- `code/`: modified harness code used for this variant.

## Key Files To Read First

1. `reports/MRPC_SAMPLING_IMPROVEMENT_REPORT.md`
2. `docs/ASSUMPTIONS_AND_SCOPE.md`
3. `docs/PIPELINE.md`
4. `docs/CHATGPT_PRO_ANALYSIS_GUIDE.md`
5. `reports/final_metrics.csv`
6. `reports/sampling_health.csv`
7. `reports/run_manifest.csv`

## W&B

Project: `lgd_bert_sampling_improvement_mrpc`

- Random baseline: https://wandb.ai/nilansa/lgd_bert_sampling_improvement_mrpc/runs/7ppykeie
- K=3: https://wandb.ai/nilansa/lgd_bert_sampling_improvement_mrpc/runs/qngb9dsj
- K=4: https://wandb.ai/nilansa/lgd_bert_sampling_improvement_mrpc/runs/0j21c5js
- K=5: https://wandb.ai/nilansa/lgd_bert_sampling_improvement_mrpc/runs/3qf9upw9
- K=6: https://wandb.ai/nilansa/lgd_bert_sampling_improvement_mrpc/runs/u18xrcc8

## Source

The variant was copied from the Sherlock audit harness and edited in a non-overwriting workspace:

- source base: `/home2/nilan/research/AN/scratch_exps/fage_sherlock_tmp/bert-lgd-reproduction/lgd_bert_sherlock_audit_20260628`
- run workspace: `/home2/nilan/research/AN/scratch_exps/bert_lgd_sherlock_sampling_improvement_20260701_1525`

W&B internal folders, model checkpoints, and unrelated historical run artifacts are not included.

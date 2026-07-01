# Manifest

This bundle contains the MRPC sampling-improvement variant from 2026-07-01.

## Included

- `README.md`: short summary and headline table.
- `docs/ASSUMPTIONS_AND_SCOPE.md`: focused assumptions and experiment scope.
- `docs/PIPELINE.md`: environment, launcher, commands, metrics, and outputs.
- `docs/CODE_CHANGES.md`: harness changes made for this variant.
- `docs/CHATGPT_PRO_ANALYSIS_GUIDE.md`: suggested review path and next questions.
- `docs/assumptions.md`: inherited reproduction assumptions from the source harness.
- `docs/sherlock_assumptions.md`: inherited Sherlock audit assumptions.
- `docs/paper_hyperparams.md`: inherited paper hyperparameter notes.
- `docs/repro_notes.md`: inherited reproduction notes.
- `reports/`: final report, metrics, health, manifest, paper notes, environment capture.
- `runs/`: per-run command/config/data/eval/train/sample logs and summaries.
- `logs/`: stdout/stderr logs from the launcher.
- `code/`: modified harness code and tests needed to inspect or rerun this exact variant.

## Excluded

- W&B internal run directories.
- model checkpoints.
- old reproduction runs unrelated to this variant.
- unrelated local untracked Sherlock audit folder in the checkout.

## Completed Runs

- `MRPC_random_epoch_shuffle_seed0`
- `MRPC_LA_K3_L50_BWoR_refreshEpoch_noCorr_seed0`
- `MRPC_LA_K4_L50_BWoR_refreshEpoch_noCorr_seed0`
- `MRPC_LA_K5_L50_BWoR_refreshEpoch_noCorr_seed0`
- `MRPC_LA_K6_L50_BWoR_refreshEpoch_noCorr_seed0`

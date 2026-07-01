# Pipeline

## Environment

The run executed on `gnode082`, a 4-GPU 2080 Ti node. The environment capture is in
`reports/env_gnode82.txt`.

Runtime used:

- Python: `/ssd_scratch/nilan/venvs/lgd-bert-sherlock-20260628/bin/python`
- Python version: 3.10.12
- Torch: 2.9.1+cu128
- CUDA visible device count: 4
- W&B project: `lgd_bert_sampling_improvement_mrpc`

## Workspace Creation

The non-overwriting workspace was:

`/home2/nilan/research/AN/scratch_exps/bert_lgd_sherlock_sampling_improvement_20260701_1525`

It was copied from:

`/home2/nilan/research/AN/scratch_exps/fage_sherlock_tmp/bert-lgd-reproduction/lgd_bert_sherlock_audit_20260628`

The workspace subfolders were:

- `code/`
- `runs/`
- `reports/`
- `logs/`
- `plots/`

## Launcher

The launcher is:

`code/scripts/launch_mrpc_sampling_improvement.py`

Responsibilities:

- detect or accept GPU ids.
- set `CUDA_VISIBLE_DEVICES` per child process.
- launch one job per GPU.
- queue K=6 until a GPU frees.
- write stdout/stderr to `logs/<run_name>.log`.
- write exact commands to `runs/<run_name>/command.txt`.
- generate final CSVs and the Markdown report.

Initial GPU assignment:

- GPU 0: `MRPC_random_epoch_shuffle_seed0`
- GPU 1: `MRPC_LA_K3_L50_BWoR_refreshEpoch_noCorr_seed0`
- GPU 2: `MRPC_LA_K4_L50_BWoR_refreshEpoch_noCorr_seed0`
- GPU 3: `MRPC_LA_K5_L50_BWoR_refreshEpoch_noCorr_seed0`
- GPU 0 after random finished: `MRPC_LA_K6_L50_BWoR_refreshEpoch_noCorr_seed0`

## Trainer Settings

All runs:

- task: `mrpc`
- epochs: `3`
- batch size: `32`
- optimizer: `adam`
- learning rate: `2e-5`
- model: `bert-base-uncased`
- seed: `0`
- `--audit_sample_coverage`
- `--log_sample_indices`

LGD runs:

- variant: `label_aware_lgd`
- `--replacement_mode batch_without_replacement`
- `--correction none`
- `--refresh_lsh every_epoch`
- `--sampler_start_step 0`
- `--sampler_warmup none`
- `--lsh_l 50`
- `--lsh_k` in `3,4,5,6`

## Logged Evidence

Per-run files:

- `command.txt`: exact launch command.
- `config.json`: resolved run configuration.
- `data_report.json`: GLUE split and label-count report.
- `train_log.csv`: per-step train and sampler metrics.
- `eval_log.csv`: evaluation metrics.
- `sample_step_log.csv`: per-step sample summary.
- `sample_coverage_by_epoch.csv`: epoch coverage summary.
- `sample_log.csv.gz`: per-sample indices and sampler metadata.
- `run_summary.json`: final status and total coverage.
- `wandb_url.txt`: W&B run URL.

Final aggregate files:

- `reports/final_metrics.csv`
- `reports/sampling_health.csv`
- `reports/run_manifest.csv`
- `reports/MRPC_SAMPLING_IMPROVEMENT_REPORT.md`

## Metrics Added Or Verified

The trainer logs:

- `train/loss`
- `eval/loss`
- `eval/accuracy`
- `eval/f1`
- `epoch`
- `global_step`
- `time/wall_clock_sec`
- `sampler/coverage_so_far`
- `sampler/unique_count_batch`
- `sampler/duplicate_count_batch`
- `sampler/duplicate_frac_batch`
- `sampler/label0_count_batch`
- `sampler/label1_count_batch`
- `sampler/bucket_size_mean`
- `sampler/bucket_size_max`
- `sampler/fallback_rate`
- `sampler/attempts_mean`
- `sampler/query_norm`

The per-sample log includes `global_step`, `epoch`, `train_index`, `label`, `bucket_size`,
`attempts`, `fallback`, `source`, `duplicate_within_batch`, and `seen_count_so_far`.

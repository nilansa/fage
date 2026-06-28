# Final Report: BERT/MRPC/RTE LGD Probability-Aware Reproduction

Generated on `gnode073` from:

`/home2/nilan/research/AN/scratch_exps/lgd_bert_probaware_repro_20260627`

## Implemented

- Clean harness under `src/lgd_bert/` with modular data loading, BERT model utilities, SimHash LSH, random/LGD samplers, probability correction, training, evaluation, and W&B logging.
- Scripts:
  - `scripts/inspect_env.py`
  - `scripts/run_lgd_bert.py`
  - `scripts/build_or_refresh_cache.py`
  - `scripts/launch_sweep.py`
  - `scripts/summarize_and_plot.py`
- Tests:
  - LSH table membership and bucket key validity
  - probability formula and monotonic SimHash collision probability
  - label-aware sign convention
  - sampler shapes and fallback
  - uniform corrected-loss identity
- Required docs:
  - `docs/paper_hyperparams.md`
  - `docs/assumptions.md`
  - `docs/repro_notes.md`

## Environment

- Host: `gnode073`
- Python: `/ssd_scratch/nilan/.venvs/cisamp-py14-topk-lite/bin/python`
- PyTorch: `2.9.1+cu128`
- Transformers: `5.8.1`
- Datasets: `4.8.5`
- W&B: `0.27.0`
- GPUs from `nvidia-smi`:
  - GPU 0: NVIDIA GeForce RTX 2080 Ti, 10823 MiB free / 11264 MiB total
  - GPU 1: NVIDIA GeForce RTX 2080 Ti, 10823 MiB free / 11264 MiB total
  - GPU 2: NVIDIA GeForce RTX 3080 Ti, 11921 MiB free / 12288 MiB total
  - GPU 3: NVIDIA GeForce RTX 2080 Ti, 10823 MiB free / 11264 MiB total
- Full env report:
  - `reports/env_report_20260627_145315.txt`
  - `reports/env_report_20260627_145315.json`

## Paper Settings Used

Paper-extracted settings and assumptions are in:

- `docs/paper_hyperparams.md`
- `docs/assumptions.md`

Important reproduction settings:

- Tasks: MRPC, RTE
- Model: BERTbase, implemented as `bert-base-uncased`
- Epochs: 3
- Batch size: 32
- Optimizer: `torch.optim.Adam`
- LR: `2e-5`, because the PDF text/render visibly shows ambiguous `2e.`
- LSH: `K=7`, `L=10`
- Hash family: SimHash / signed random projection
- Main representation: final hidden state for first token, `last_hidden_state[:, 0, :]`
- Refresh: `--refresh_lsh every_epoch`, because the supplement says periodic refresh but gives no exact period
- GLUE validation is logged as `paper_fig5/test_*`, because labeled GLUE test labels are unavailable

## Commands Run

```bash
cd /home2/nilan/research/AN/scratch_exps/lgd_bert_probaware_repro_20260627
export PYTHONPATH=src

/ssd_scratch/nilan/.venvs/cisamp-py14-topk-lite/bin/python -m compileall -q src scripts tests
/ssd_scratch/nilan/.venvs/cisamp-py14-topk-lite/bin/python scripts/inspect_env.py
/ssd_scratch/nilan/.venvs/cisamp-py14-topk-lite/bin/python -m pytest -q tests

CUDA_VISIBLE_DEVICES=0 /ssd_scratch/nilan/.venvs/cisamp-py14-topk-lite/bin/python scripts/run_lgd_bert.py \
  --task mrpc --variant random --correction none --seed 0 --max_steps 2 --eval_every 1 --batch_size 8 \
  --wandb_project lgd_neurips2019_bert_repro --wandb_group quick_probe_20260627 \
  --output_root runs/quick_probe_20260627

/ssd_scratch/nilan/.venvs/cisamp-py14-topk-lite/bin/python scripts/launch_sweep.py \
  --stage smoke --wandb_group smoke_20260627_1455

/ssd_scratch/nilan/.venvs/cisamp-py14-topk-lite/bin/python scripts/launch_sweep.py \
  --stage sanity --wandb_group sanity_20260627_1458

/ssd_scratch/nilan/.venvs/cisamp-py14-topk-lite/bin/python scripts/launch_sweep.py \
  --stage full_pair --allow_full --wandb_group full_pair_mrpc_seed0_20260627_1501

/ssd_scratch/nilan/.venvs/cisamp-py14-topk-lite/bin/python scripts/launch_sweep.py \
  --stage full_sweep --allow_full --seeds 0,1,2 --wandb_group full_sweep_20260627_1504

/ssd_scratch/nilan/.venvs/cisamp-py14-topk-lite/bin/python scripts/summarize_and_plot.py \
  --run_root runs/full_sweep_20260627_1504 --out_dir plots/full_sweep_20260627_1504
```

## W&B

- Project: https://wandb.ai/nilansa/lgd_neurips2019_bert_repro
- Quick probe run: https://wandb.ai/nilansa/lgd_neurips2019_bert_repro/runs/960ybii5
- Full per-run W&B links are recorded in:
  - `plots/full_sweep_20260627_1504/final_metrics.csv`
  - each run's `wandb_url.txt`

## Final Aggregate Metrics

Mean over seeds 0, 1, 2 from `plots/full_sweep_20260627_1504/final_metrics.csv`.

| Task | Variant | N | Final accuracy mean | Per-seed accuracy | Final loss mean | Final F1 mean |
|---|---|---:|---:|---|---:|---:|
| MRPC | SGD/random | 3 | 0.8309 | 0.8333, 0.8333, 0.8260 | 0.4148 | 0.8797 |
| MRPC | paper LGD | 3 | 0.6797 | 0.7010, 0.7500, 0.5882 | 1.1185 | 0.7607 |
| MRPC | paper LGD full | 3 | 0.7239 | 0.7328, 0.7181, 0.7206 | 0.7420 | 0.8113 |
| MRPC | label-aware LGD | 3 | 0.5743 | 0.6838, 0.3162, 0.7230 | 2.3698 | 0.5354 |
| MRPC | label-aware LGD corrected | 3 | 0.6904 | 0.4877, 0.8137, 0.7696 | 0.6837 | 0.7499 |
| RTE | SGD/random | 3 | 0.6534 | 0.6751, 0.6245, 0.6606 | 0.8776 | 0.6636 |
| RTE | paper LGD | 3 | 0.5800 | 0.6173, 0.5848, 0.5379 | 1.2055 | 0.5432 |
| RTE | paper LGD full | 3 | 0.5788 | 0.5740, 0.5704, 0.5921 | 1.0538 | 0.4245 |
| RTE | label-aware LGD | 3 | 0.5439 | 0.5343, 0.5451, 0.5523 | 1.5162 | 0.4047 |
| RTE | label-aware LGD corrected | 3 | 0.5066 | 0.5054, 0.5162, 0.4982 | 2.0796 | 0.3366 |

## Probability Correction And Fallback

- Exact full probability correction was used for all `correction=full` runs.
- No clipped mode was launched because full correction did not produce NaNs or crashes.
- `correction/weight_clipped_frac` stayed `0.0` in all runs because clipping was disabled for exact full correction.
- Maximum observed full-correction weight over full training logs:
  - MRPC paper LGD full: 45.5471
  - MRPC label-aware LGD corrected: 9.6622
  - RTE paper LGD full: 24.0150
  - RTE label-aware LGD corrected: 8.9569
- Fallback:
  - RTE LGD runs: max fallback rate 0.0
  - MRPC paper LGD runs: max fallback rate 0.0
  - MRPC label-aware LGD none/full: some seeds reached fallback rate 1.0 on logged steps; max mean fallback over logged steps was 0.3333.

## Figure-5-Style Plots

Exported local PNGs:

- `plots/full_sweep_20260627_1504/mrpc_accuracy_vs_iter.png`
- `plots/full_sweep_20260627_1504/rte_accuracy_vs_iter.png`
- `plots/full_sweep_20260627_1504/mrpc_loss_vs_iter.png`
- `plots/full_sweep_20260627_1504/rte_loss_vs_iter.png`
- `plots/full_sweep_20260627_1504/figure5_bert_mrpc_rte_accuracy_loss.png`

## Deviations From The Paper

- The learning-rate exponent is ambiguous in the provided PDF; main runs used `2e-5`.
- The checkpoint name is not specified by the paper; main runs used `bert-base-uncased`.
- GLUE validation is used as the labeled "testing" curve.
- Evaluation frequency is every 25 steps plus final step; the paper does not specify frequency.
- Adam betas, epsilon, weight decay, warmup, schedule, max sequence length, and gradient clipping are not specified by the paper; defaults are documented in `docs/assumptions.md`.
- The binary classifier query mapping is an implementation interpretation of "classification-layer parameters are used as queries."
- The label-aware variant is a requested extension based on the logistic LGD derivation; the BERT supplement itself only says pooled representations are placed in LSH tables.


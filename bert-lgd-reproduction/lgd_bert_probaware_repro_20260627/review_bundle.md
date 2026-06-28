# Review Bundle: BERT LGD Probability-Aware Reproduction

This generated file concatenates the core README, docs, reports, source, scripts, and tests for review.



---

## `README.md`

```markdown
# BERT LGD Probability-Aware Reproduction

This is a self-contained reproduction harness for the NeurIPS 2019 LGD paper's BERT/MRPC/RTE experiment, plus two extensions requested during the run:

- paper-style BERT LGD sampling
- label-aware BERT LGD sampling
- exact probability correction using `1 / (N * p_i)`
- optional biased/stability correction modes such as clipped, sqrt, and normalized weights

The code was built and run on `gnode073` with 4 GPUs. The final full sweep completed 30 runs across MRPC/RTE, seeds `0,1,2`, and the random, paper-style LGD, and label-aware LGD variants.

## Main Question This Repo Supports

The goal is to let another model or reviewer inspect:

1. Whether the BERT LGD implementation follows the paper/supplement closely.
2. Whether the probability formula and corrected-loss weighting are implemented correctly.
3. Why the reproduced LGD variants did not beat the random baseline in these runs.
4. What implementation or experimental changes should be tried next.

## Paper And Supplement Interpretation

Primary local paper files used during the run:

- `/home2/nilan/research/AN/scratch_exps/papers/Experiments-NeurIPS-2019-fast-and-accurate-stochastic-gradient-estimation-Paper.pdf`
- `/home2/nilan/research/AN/scratch_exps/papers/experiments-lgd_supplement.pdf`

Extracted hard settings:

- Tasks: MRPC and RTE
- Model: BERTbase
- Epochs: 3
- Batch size: 32
- Optimizer: Adam
- LSH parameters: `K=7`, `L=10`
- Plots: testing/validation accuracy and loss vs iteration
- Stored BERT representation: final hidden state for the first token, implemented as `last_hidden_state[:, 0, :]`

Ambiguous or assumed settings are documented in `docs/assumptions.md`. The most important one is learning rate: the rendered PDF showed `2e.` with the exponent missing, so the main runs used `2e-5`.

## Sampling Variants

Random baseline:

```text
p_i = 1 / N
loss = mean(CE_i)
```

Paper-style non-label-aware BERT LGD:

```text
stored v_i = h_i
theta = W[1] - W[0]
query q = theta
```

Label-aware BERT LGD:

```text
label 1 -> ytilde = +1
label 0 -> ytilde = -1
stored v_i = ytilde_i * h_i
theta = W[1] - W[0]
query q = -theta = W[0] - W[1]
```

The label-aware version follows the paper's logistic-regression derivation, with the BERT CLS representation `h_i` replacing `x_i`.

## Probability Formula

For a selected sample `i`, query `q`, stored vector `v_i`, bucket size `S`, and table-attempt count `l`:

```text
cos_i = dot(v_i, q) / (||v_i|| ||q||)
cp_i = 1 - arccos(cos_i) / pi
cpK_i = cp_i ** K
p_i = cpK_i * (1 - cpK_i) ** (l - 1) * (1 / S)
```

The corrected loss is:

```python
losses = F.cross_entropy(logits, labels, reduction="none")
weights = 1.0 / (N * p_i)
loss = (weights * losses).mean()
```

For uniform sampling, `p_i = 1/N`, so `weights = 1` and the corrected loss reduces to ordinary mean cross-entropy.

## Batch Size 32

LGD samples one example per draw, but a BERT optimizer step uses a minibatch of 32 independent draws:

```text
draw 32 indices from the sampler
keep each sample's p_i
forward all 32 tokenized examples through BERT together
compute 32 per-example CE losses
apply per-example correction weights if enabled
average and backprop once
```

Duplicates are allowed because the harness uses repeated independent single-sample draws. See `reports/STATIC_ANALYSIS_LSH_SAMPLING.md`.

## LSH Refresh

All full LGD runs used:

```text
--refresh_lsh every_epoch
```

The harness recomputes CLS representations and rebuilds LSH tables before training and after each epoch boundary.

- MRPC: initial build, then refresh at steps `116` and `231`.
- RTE: initial build, then refresh at steps `79` and `157`.
- Random baseline: no LSH build or refresh.

## Execution Protocol

The actual run sequence was:

1. Paper/supplement inspection and documentation.
2. Environment inspection.
3. Unit tests.
4. 20-step smoke tests.
5. 100-step sanity tests.
6. One full MRPC random baseline plus one full MRPC label-aware corrected run.
7. Full MRPC/RTE multi-seed sweep.

Main environment:

```bash
export PYTHONPATH=src
/ssd_scratch/nilan/.venvs/cisamp-py14-topk-lite/bin/python
```

## Important Results

Final mean validation accuracy over seeds `0,1,2`:

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

In these runs, random/SGD was stronger than LGD. That mismatch from the paper is the main thing to investigate next.

## Files To Read First

- `reports/FINAL_REPORT.md`
- `reports/STATIC_ANALYSIS_LSH_SAMPLING.md`
- `docs/paper_hyperparams.md`
- `docs/assumptions.md`
- `src/lgd_bert/samplers.py`
- `src/lgd_bert/probability.py`
- `src/lgd_bert/train.py`
- `plots/full_sweep_20260627_1504/final_metrics.csv`
- `plots/full_sweep_20260627_1504/eval_curves.csv`

## Known Limitations

- The full-run logs did not store every sampled index or every individual `p_i`; they stored aggregate sampler stats. The code kept per-sample probabilities during training, but exact duplicate rates from finished runs cannot be reconstructed.
- The paper did not specify some BERT implementation details: checkpoint cased/uncased, LR exponent, Adam betas/epsilon, schedule, warmup, max sequence length, exact refresh period, and evaluation interval.
- GLUE validation is used as the paper-style "testing" curve because labeled GLUE test labels are not locally available.
- The binary classifier query mapping is an implementation interpretation of the supplement's statement that classifier-layer parameters are used as queries.
```


---

## `docs/assumptions.md`

```markdown
# Assumptions And Deviations

| Topic | Assumption / deviation | Reason |
|---|---|---|
| Learning rate | Main runs use `2e-5`. | The paper text and rendered page show `2e.` with no visible exponent. `2e-5` is the standard BERT GLUE fine-tuning value and was requested as the preferred assumption if the PDF remains ambiguous. |
| Checkpoint | Main runs use `bert-base-uncased`. | The paper says BERTbase but does not name cased vs uncased. |
| GLUE test split | The harness logs HF GLUE validation as `paper_fig5/test_*`. | Standard GLUE test labels are not available locally through `datasets`; validation is labeled. |
| HF dataset sizes | The harness records actual HF train/validation sizes in each run's `data_report.json`. | The paper table gives approximate/paper counts; local HF GLUE may differ by one example from the paper table. |
| Loss function | Main runs use `F.cross_entropy(logits, labels, reduction="none")`. | The BERT section does not explicitly name the loss; this is the standard sequence-classification objective. |
| Optimizer details | Main runs use `torch.optim.Adam` with betas `(0.9, 0.999)`, epsilon `1e-8`, no weight decay, no warmup, and no schedule. | The paper says Adam but does not specify betas, epsilon, weight decay, warmup, or schedule. AdamW is not used in the main reproduction. |
| Max sequence length | Main runs use `128`. | The paper says it replicated BERT paper settings but does not specify this detail in the provided PDFs. |
| Representation refresh | Main runs default to `--refresh_lsh every_epoch`. | The supplement says representations can be periodically updated but gives no exact period. `initial_only` is available as an ablation. |
| BERT representation | Main runs use `outputs.last_hidden_state[:, 0, :]` from the base BERT model. | The supplement says final hidden state for the first token. `pooler_output` is not used for the main reproduction. |
| Non-label-aware query | `q = W[1] - W[0]`, stored `v_i = h_i`. | The supplement says classifier-layer parameters are queries but does not spell out binary-softmax reduction. |
| Label-aware query | `q = W[0] - W[1]`, stored `v_i = ytilde_i * h_i`. | This follows the paper's logistic LGD derivation with `h_i` replacing `x_i`. |
| Bias term | Bias is disabled in the main run and available as `--include_bias_in_lsh`. | The supplement says fixed-dimensional pooled representations are stored in LSH; it does not mention appending bias. |
| Probability correction | `full` mode uses weights `1 / (N * p_i)`. | This implements the requested probability-aware LGD estimator. `clipped`, `sqrt`, and `normalized` are marked as biased/stability ablations. |
| Probability formula | For selected sample `i`, `p_i = cpK_i * (1 - cpK_i)^(l-1) * (1 / S)`. | The paper excerpt states the collision part; the harness includes the uniform draw from bucket size `S` as requested. |
| Fallback | If all queried tables are empty, fallback samples uniformly with `p_i=1/N`. | Required to avoid invalid probabilities and logged as `sampler/fallback_rate`. |

```


---

## `docs/paper_hyperparams.md`

```markdown
# Paper Hyperparameters

Sources inspected:

- `/home2/nilan/research/AN/scratch_exps/papers/Experiments-NeurIPS-2019-fast-and-accurate-stochastic-gradient-estimation-Paper.pdf`
- `/home2/nilan/research/AN/scratch_exps/papers/experiments-lgd_supplement.pdf`

The main PDF is a three-page excerpt covering the experiments section. `pdftotext` was used first, and page 3 was rendered with `pdftoppm` to inspect the ambiguous BERT learning-rate line.

| Field | MRPC | RTE | Source / note |
|---|---:|---:|---|
| task | MRPC | RTE | Main paper section 3.2 and Figure 5 |
| model | BERTbase | BERTbase | Main paper section 3.2 |
| pretrained checkpoint name | assumed `bert-base-uncased` | assumed `bert-base-uncased` | Paper says BERTbase but not cased/uncased |
| train size | 3669 in paper table | 2491 in paper table | Main paper Figure 4 table |
| validation/test size | 409 in paper table | 278 in paper table | Main paper Figure 4 table |
| number of epochs | 3 | 3 | Main paper section 3.2 |
| batch size | 32 | 32 | Main paper section 3.2 |
| optimizer | Adam | Adam | Main paper section 3.2 |
| learning rate | ambiguous `2e.`; harness default `2e-5` | ambiguous `2e.`; harness default `2e-5` | Rendered PDF page still shows no exponent |
| loss function | per-example classification cross entropy | per-example classification cross entropy | Standard BERT binary sequence-classification assumption |
| LSH K | 7 | 7 | Main paper section 3.2 |
| LSH L | 10 | 10 | Main paper section 3.2 |
| hash family | SimHash / signed random projection | SimHash / signed random projection | Main paper linear-regression LSH details; BERT section only gives K/L |
| representation stored in LSH | final hidden state for first token | final hidden state for first token | Supplement section 5 |
| classifier query definition | classification-layer parameters | classification-layer parameters | Supplement section 5; binary-softmax query mapping is an implementation assumption |
| pooled-representation refresh rule | periodically updated | periodically updated | Supplement section 5; no exact period found |
| evaluation frequency | not specified | not specified | Harness default is every 25 steps and final step |
| plotted metrics | testing accuracy and testing loss vs iteration | testing accuracy and testing loss vs iteration | Main paper Figure 5 |
| comparison | SGD/random vs LGD | SGD/random vs LGD | Main paper Figure 5 |

Algorithm details verified from the provided paper excerpt:

- The probability text states `p_i = cp(x_i, theta_t)^K * (1 - cp(x_i, theta_t)^K)^(l-1)`, where `l` is the number of tables used by the sampling process.
- The logistic-regression paragraph states that labels are in `{-1,+1}`, preprocess `y_i * x_i`, and query with `-theta_t`.
- The requested probability-aware harness uses the standard importance correction `gradient / (N * p_i)`, implemented through per-sample cross-entropy weights `1 / (N * p_i)`.

```


---

## `docs/repro_notes.md`

```markdown
# Reproduction Notes

Execution order for this folder:

1. Inspect environment:

   ```bash
   PYTHONPATH=src /ssd_scratch/nilan/.venvs/cisamp-py14-topk-lite/bin/python scripts/inspect_env.py
   ```

2. Run unit tests:

   ```bash
   PYTHONPATH=src /ssd_scratch/nilan/.venvs/cisamp-py14-topk-lite/bin/python -m pytest -q tests
   ```

3. Run 20-step smoke tests:

   ```bash
   PYTHONPATH=src /ssd_scratch/nilan/.venvs/cisamp-py14-topk-lite/bin/python scripts/launch_sweep.py --stage smoke
   ```

4. Run 100-step sanity tests:

   ```bash
   PYTHONPATH=src /ssd_scratch/nilan/.venvs/cisamp-py14-topk-lite/bin/python scripts/launch_sweep.py --stage sanity
   ```

5. Run the required full MRPC pair:

   ```bash
   PYTHONPATH=src /ssd_scratch/nilan/.venvs/cisamp-py14-topk-lite/bin/python scripts/launch_sweep.py --stage full_pair --allow_full
   ```

6. Launch full sweep only after the previous gates pass:

   ```bash
   PYTHONPATH=src /ssd_scratch/nilan/.venvs/cisamp-py14-topk-lite/bin/python scripts/launch_sweep.py --stage full_sweep --allow_full --seeds 0,1,2
   ```

Main W&B project:

```bash
WANDB_PROJECT=lgd_neurips2019_bert_repro
```

Large model and dataset caches use `/ssd_scratch/nilan/.cache/huggingface` by default.

```


---

## `reports/FINAL_REPORT.md`

```markdown
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

```


---

## `reports/STATIC_ANALYSIS_LSH_SAMPLING.md`

```markdown
# Static Analysis: LSH Sampling And Pooled-Representation Refresh

Scope:

- Full runs under `runs/full_sweep_20260627_1504`
- Code under `src/lgd_bert/`

## Does the same sample get sampled from LSH tables?

Yes, duplicate sampling is allowed by the current LGD sampler.

Code evidence:

- `LGDBatchSampler.sample_batch()` builds a batch as repeated independent calls to `sample_one()`:
  - `rows = [self.sample_one(query) for _ in range(int(batch_size))]`
- `sample_one()` chooses one index from the returned bucket:
  - `index = int(self.rng.choice(bucket))`
- There is no `selected_set`, `avoid_indices`, `without_replacement`, or historical exclusion state in `LGDBatchSampler`.

Implication:

- The same train index can appear more than once in a single mini-batch.
- The same train index can also be sampled repeatedly across different steps.
- This matches the intended probability-aware setup where the batch is formed by repeated independent single-sample draws, but it means there is no no-duplicate guarantee.

Important logging limitation:

- The full-run `train_log.csv` files do not contain `sample.indices`.
- They only contain aggregate fields such as `sampler/p_mean`, `sampler/bucket_size_mean`, `sampler/attempts_mean`, and `sampler/fallback_rate`.
- Therefore the exact duplicate rate in the already-completed full runs cannot be computed after the fact from the current logs.

To quantify this in a future run, add per-step logging for:

- sampled indices
- unique count per batch
- duplicate count per batch
- top-k most frequently sampled indices per run
- whether each index came from LSH or fallback

## LSH table membership

Each stored train index is inserted once per table when an LSH index is built.

Code evidence:

- `SimHashLSH.fit()` resets all tables.
- For each stored vector and train index, it loops over all tables and appends the index to one bucket in each table.
- Therefore, with `L=10`, each train example appears exactly once in each of the 10 tables, but can be retrieved repeatedly by independent queries.

## How often were pooled representations updated?

All full LGD runs used:

```text
refresh_lsh = every_epoch
```

The implementation performs:

1. Initial CLS representation computation and LSH build before training.
2. Refresh at the first step after each epoch boundary.

Refresh code path:

- `build_lsh_state()` recomputes CLS representations with the current BERT model.
- It rebuilds stored vectors according to the variant.
- It rebuilds SimHash tables.
- `sampler.update_index()` swaps the sampler to the new LSH index.

For MRPC:

- train size: `3668`
- batch size: `32`
- steps per epoch: `115`
- total steps: `345`
- LSH builds per LGD run: `3`
  - initial build before step 1
  - refresh at step `116`
  - refresh at step `231`

For RTE:

- train size: `2490`
- batch size: `32`
- steps per epoch: `78`
- total steps: `234`
- LSH builds per LGD run: `3`
  - initial build before step 1
  - refresh at step `79`
  - refresh at step `157`

Random baseline runs do not build or refresh LSH:

- `lsh/refresh_count = 0`

## Fallback observations from full logs

From aggregate full-run logs:

- RTE LGD runs: fallback max `0.0`
- MRPC paper LGD runs: fallback max `0.0`
- MRPC label-aware LGD runs: some seeds reached logged fallback rate `1.0`; max mean fallback over logged steps was `0.3333`

This fallback statistic is aggregate per step; it does not reveal which exact indices were sampled.

```


---

## `src/lgd_bert/__init__.py`

```python
"""BERT/MRPC/RTE LGD reproduction harness."""

```


---

## `src/lgd_bert/config.py`

```python
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER_MAIN = Path("/home2/nilan/research/AN/scratch_exps/papers/Experiments-NeurIPS-2019-fast-and-accurate-stochastic-gradient-estimation-Paper.pdf")
PAPER_SUPPLEMENT = Path("/home2/nilan/research/AN/scratch_exps/papers/experiments-lgd_supplement.pdf")
DEFAULT_WANDB_PROJECT = "lgd_neurips2019_bert_repro"
DEFAULT_MODEL = "bert-base-uncased"


@dataclass
class TrainConfig:
    task: str = "mrpc"
    variant: str = "random"
    correction: str = "none"
    seed: int = 0
    model_name: str = DEFAULT_MODEL
    batch_size: int = 32
    epochs: int = 3
    lr: float = 2e-5
    optimizer: str = "adam"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    max_seq_length: int = 128
    eval_every: int = 25
    log_every: int = 1
    max_steps: int | None = None
    lsh_k: int = 7
    lsh_l: int = 10
    hash_family: str = "simhash_signed_random_projection"
    refresh_lsh: str = "every_epoch"
    refresh_steps: int = 100
    representation: str = "cls_last_hidden_state"
    include_bias_in_lsh: bool = False
    correction_max_weight: float = 10.0
    probability_eps: float = 1e-12
    freeze_bert: bool = False
    max_grad_norm: float = 0.0
    wandb_project: str = DEFAULT_WANDB_PROJECT
    wandb_mode: str = "online"
    wandb_group: str | None = None
    run_name: str | None = None
    run_dir: str | None = None
    output_root: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["paper_main"] = str(PAPER_MAIN)
        data["paper_supplement"] = str(PAPER_SUPPLEMENT)
        return data

```


---

## `src/lgd_bert/data.py`

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset


TASK_TO_KEYS = {
    "mrpc": ("sentence1", "sentence2"),
    "rte": ("sentence1", "sentence2"),
}


@dataclass
class EncodedDataset:
    task: str
    split: str
    features: dict[str, torch.Tensor]
    labels: torch.Tensor
    raw_size: int

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def batch(self, indices: list[int] | torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
        if not torch.is_tensor(indices):
            indices = torch.tensor(indices, dtype=torch.long)
        batch = {key: value[indices].to(device) for key, value in self.features.items()}
        batch["labels"] = self.labels[indices].to(device)
        return batch


@dataclass
class GlueData:
    train: EncodedDataset
    validation: EncodedDataset
    report: dict


def _encode_split(raw_split, tokenizer, task: str, split: str, max_length: int) -> EncodedDataset:
    key1, key2 = TASK_TO_KEYS[task]
    encoded = tokenizer(
        list(raw_split[key1]),
        list(raw_split[key2]),
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    keep_keys = ["input_ids", "attention_mask", "token_type_ids"]
    features = {
        key: torch.tensor(encoded[key], dtype=torch.long)
        for key in keep_keys
        if key in encoded
    }
    labels = torch.tensor(raw_split["label"], dtype=torch.long)
    return EncodedDataset(task=task, split=split, features=features, labels=labels, raw_size=len(raw_split))


def load_glue_data(task: str, tokenizer, max_length: int = 128) -> GlueData:
    task = task.lower()
    if task not in TASK_TO_KEYS:
        raise ValueError(f"Unsupported GLUE task: {task}")
    try:
        raw = load_dataset("glue", task)
        source = "Hugging Face datasets: glue/" + task
    except Exception:
        raw = load_dataset("nyu-mll/glue", task)
        source = "Hugging Face datasets: nyu-mll/glue/" + task
    train = _encode_split(raw["train"], tokenizer, task, "train", max_length)
    validation = _encode_split(raw["validation"], tokenizer, task, "validation", max_length)
    train_counts = torch.bincount(train.labels, minlength=2).tolist()
    val_counts = torch.bincount(validation.labels, minlength=2).tolist()
    paper_expected = {
        "mrpc": {"train": 3669, "validation_or_test": 409},
        "rte": {"train": 2491, "validation_or_test": 278},
    }[task]
    report = {
        "task": task,
        "source": source,
        "train_size": len(train),
        "validation_size": len(validation),
        "train_label_counts": train_counts,
        "validation_label_counts": val_counts,
        "paper_expected_approx": paper_expected,
        "testing_curve_split": "validation",
        "note": "GLUE test labels are unavailable through the standard dataset, so validation is logged as paper_fig5/test_*.",
    }
    return GlueData(train=train, validation=validation, report=report)


def save_data_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
```


---

## `src/lgd_bert/eval.py`

```python
from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import EncodedDataset


def binary_f1(preds: list[int], labels: list[int]) -> float:
    tp = sum(1 for pred, label in zip(preds, labels) if pred == 1 and label == 1)
    fp = sum(1 for pred, label in zip(preds, labels) if pred == 1 and label == 0)
    fn = sum(1 for pred, label in zip(preds, labels) if pred == 0 and label == 1)
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else float(2 * tp / denom)


@torch.no_grad()
def evaluate(model, dataset: EncodedDataset, batch_size: int, device: torch.device) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_count = 0
    preds: list[int] = []
    labels_all: list[int] = []
    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        batch = dataset.batch(indices, device)
        labels = batch.pop("labels")
        outputs = model(**batch)
        logits = outputs.logits
        losses = F.cross_entropy(logits, labels, reduction="none")
        total_loss += float(losses.detach().sum().cpu())
        total_count += int(labels.shape[0])
        preds.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
        labels_all.extend(labels.detach().cpu().tolist())
    if was_training:
        model.train()
    correct = sum(1 for pred, label in zip(preds, labels_all) if pred == label)
    return {
        "eval/loss": total_loss / max(total_count, 1),
        "eval/accuracy": correct / max(total_count, 1),
        "eval/f1": binary_f1(preds, labels_all),
    }

```


---

## `src/lgd_bert/lsh.py`

```python
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class LSHSummary:
    table_count: int
    k: int
    dim: int
    nonempty_bucket_mean: float
    nonempty_bucket_max: int
    nonempty_bucket_min: int


class SimHashLSH:
    def __init__(self, dim: int, k: int = 7, l: int = 10, seed: int = 0) -> None:
        self.dim = int(dim)
        self.k = int(k)
        self.l = int(l)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        projections = self.rng.normal(size=(self.l, self.k, self.dim)).astype("float32")
        projections /= np.linalg.norm(projections, axis=2, keepdims=True) + 1e-12
        self.projections = projections
        self.tables: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(self.l)]

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype="float32")
        return vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-12)

    def hash_vector(self, vector: np.ndarray, table_id: int) -> int:
        row = self._normalize(np.asarray(vector, dtype="float32").reshape(1, -1))[0]
        bits = (self.projections[int(table_id)] @ row >= 0).astype(np.uint8)
        key = 0
        for bit in bits:
            key = (key << 1) | int(bit)
        return int(key)

    def fit(self, vectors: np.ndarray, indices: np.ndarray | None = None) -> None:
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected vectors with shape [n,{self.dim}], got {vectors.shape}")
        if indices is None:
            indices = np.arange(vectors.shape[0])
        indices = np.asarray(indices, dtype=np.int64)
        if indices.shape[0] != vectors.shape[0]:
            raise ValueError("indices length must match vectors length")
        self.tables = [defaultdict(list) for _ in range(self.l)]
        normalized = self._normalize(vectors)
        for row, index in zip(normalized, indices):
            for table_id in range(self.l):
                key = self.hash_vector(row, table_id)
                self.tables[table_id][key].append(int(index))

    def query_bucket(self, query: np.ndarray, table_id: int) -> tuple[int, list[int]]:
        key = self.hash_vector(query, table_id)
        return key, list(self.tables[int(table_id)].get(key, []))

    def all_bucket_sizes(self) -> list[int]:
        return [len(bucket) for table in self.tables for bucket in table.values()]

    def summary(self) -> LSHSummary:
        sizes = self.all_bucket_sizes()
        return LSHSummary(
            table_count=self.l,
            k=self.k,
            dim=self.dim,
            nonempty_bucket_mean=float(np.mean(sizes)) if sizes else 0.0,
            nonempty_bucket_max=int(max(sizes)) if sizes else 0,
            nonempty_bucket_min=int(min(sizes)) if sizes else 0,
        )

```


---

## `src/lgd_bert/model_utils.py`

```python
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

```


---

## `src/lgd_bert/probability.py`

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ProbabilityStats:
    weights: torch.Tensor
    weight_mean: float
    weight_min: float
    weight_max: float
    weight_clipped_frac: float


def cosine_similarity(vector: np.ndarray, query: np.ndarray, eps: float = 1e-12) -> float:
    v = np.asarray(vector, dtype="float32")
    q = np.asarray(query, dtype="float32")
    denom = (float(np.linalg.norm(v)) * float(np.linalg.norm(q))) + eps
    return float(np.dot(v, q) / denom)


def simhash_collision_probability(cosine: float, eps: float = 1e-7) -> float:
    clipped = float(np.clip(cosine, -1.0 + eps, 1.0 - eps))
    return float(1.0 - np.arccos(clipped) / np.pi)


def lsh_sampling_probability(
    vector: np.ndarray,
    query: np.ndarray,
    k: int,
    bucket_size: int,
    attempts: int,
    eps: float = 1e-12,
) -> tuple[float, float, float, float, bool]:
    if bucket_size <= 0 or attempts <= 0:
        raise ValueError("bucket_size and attempts must be positive")
    cos = cosine_similarity(vector, query)
    cp = simhash_collision_probability(cos)
    cp_k = float(cp**k)
    p = float(cp_k * ((1.0 - cp_k) ** (attempts - 1)) * (1.0 / bucket_size))
    clamped = p < eps
    return float(max(p, eps)), cp, cp_k, cos, clamped


def corrected_loss_weights(
    probabilities: torch.Tensor,
    train_size: int,
    mode: str,
    max_weight: float = 10.0,
    eps: float = 1e-12,
) -> ProbabilityStats:
    if mode == "none":
        weights = torch.ones_like(probabilities)
        clipped_frac = 0.0
    else:
        raw = 1.0 / (float(train_size) * torch.clamp(probabilities, min=eps))
        if mode == "full":
            weights = raw
            clipped_frac = 0.0
        elif mode == "clipped":
            weights = torch.clamp(raw, max=max_weight)
            clipped_frac = float((raw > max_weight).float().mean().detach().cpu())
        elif mode == "sqrt":
            weights = torch.sqrt(raw)
            clipped_frac = 0.0
        elif mode == "normalized":
            weights = raw / torch.clamp(raw.mean(), min=eps)
            clipped_frac = 0.0
        else:
            raise ValueError(f"Unsupported correction mode: {mode}")
    detached = weights.detach().float().cpu()
    return ProbabilityStats(
        weights=weights,
        weight_mean=float(detached.mean()),
        weight_min=float(detached.min()),
        weight_max=float(detached.max()),
        weight_clipped_frac=clipped_frac,
    )

```


---

## `src/lgd_bert/samplers.py`

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lsh import SimHashLSH
from .probability import lsh_sampling_probability


@dataclass
class SampleBatch:
    indices: list[int]
    probabilities: list[float]
    collision_probs: list[float]
    collision_probs_k: list[float]
    cosines: list[float]
    bucket_sizes: list[int]
    attempts: list[int]
    fallback_flags: list[bool]
    probability_clamped_flags: list[bool]

    def stats(self) -> dict[str, float]:
        p = np.asarray(self.probabilities, dtype="float64")
        cp = np.asarray(self.collision_probs, dtype="float64") if self.collision_probs else np.asarray([0.0])
        cp_k = np.asarray(self.collision_probs_k, dtype="float64") if self.collision_probs_k else np.asarray([0.0])
        buckets = np.asarray([size for size in self.bucket_sizes if size > 0], dtype="float64")
        attempts = np.asarray(self.attempts, dtype="float64")
        fallbacks = np.asarray(self.fallback_flags, dtype="float64")
        return {
            "sampler/p_mean": float(p.mean()),
            "sampler/p_min": float(p.min()),
            "sampler/p_max": float(p.max()),
            "sampler/cp_mean": float(cp.mean()),
            "sampler/cpK_mean": float(cp_k.mean()),
            "sampler/bucket_size_mean": float(buckets.mean()) if buckets.size else 0.0,
            "sampler/bucket_size_max": float(buckets.max()) if buckets.size else 0.0,
            "sampler/attempts_mean": float(attempts.mean()) if attempts.size else 0.0,
            "sampler/fallback_rate": float(fallbacks.mean()) if fallbacks.size else 0.0,
            "sampler/probability_clamped_frac": float(np.mean(self.probability_clamped_flags)) if self.probability_clamped_flags else 0.0,
        }


class RandomBatchSampler:
    def __init__(self, train_size: int, seed: int = 0) -> None:
        self.train_size = int(train_size)
        self.rng = np.random.default_rng(seed)

    def sample_batch(self, batch_size: int) -> SampleBatch:
        indices = self.rng.integers(0, self.train_size, size=int(batch_size)).astype(int).tolist()
        return SampleBatch(
            indices=indices,
            probabilities=[1.0 / self.train_size] * len(indices),
            collision_probs=[0.0] * len(indices),
            collision_probs_k=[0.0] * len(indices),
            cosines=[0.0] * len(indices),
            bucket_sizes=[self.train_size] * len(indices),
            attempts=[1] * len(indices),
            fallback_flags=[False] * len(indices),
            probability_clamped_flags=[False] * len(indices),
        )


class LGDBatchSampler:
    def __init__(
        self,
        lsh: SimHashLSH,
        stored_vectors: np.ndarray,
        train_size: int,
        seed: int = 0,
        probability_eps: float = 1e-12,
    ) -> None:
        self.lsh = lsh
        self.stored_vectors = np.asarray(stored_vectors, dtype="float32")
        self.train_size = int(train_size)
        self.rng = np.random.default_rng(seed)
        self.probability_eps = float(probability_eps)

    def update_index(self, lsh: SimHashLSH, stored_vectors: np.ndarray) -> None:
        self.lsh = lsh
        self.stored_vectors = np.asarray(stored_vectors, dtype="float32")

    def sample_one(self, query: np.ndarray) -> tuple[int, float, float, float, float, int, int, bool, bool]:
        table_order = self.rng.permutation(self.lsh.l)
        for attempts, table_id in enumerate(table_order, start=1):
            _key, bucket = self.lsh.query_bucket(query, int(table_id))
            if not bucket:
                continue
            index = int(self.rng.choice(bucket))
            bucket_size = int(len(bucket))
            p, cp, cp_k, cos, clamped = lsh_sampling_probability(
                self.stored_vectors[index],
                query,
                self.lsh.k,
                bucket_size,
                attempts,
                eps=self.probability_eps,
            )
            return index, p, cp, cp_k, cos, bucket_size, attempts, False, clamped
        index = int(self.rng.integers(0, self.train_size))
        return index, 1.0 / self.train_size, 0.0, 0.0, 0.0, self.train_size, self.lsh.l, True, False

    def sample_batch(self, batch_size: int, query: np.ndarray) -> SampleBatch:
        rows = [self.sample_one(query) for _ in range(int(batch_size))]
        return SampleBatch(
            indices=[row[0] for row in rows],
            probabilities=[row[1] for row in rows],
            collision_probs=[row[2] for row in rows],
            collision_probs_k=[row[3] for row in rows],
            cosines=[row[4] for row in rows],
            bucket_sizes=[row[5] for row in rows],
            attempts=[row[6] for row in rows],
            fallback_flags=[row[7] for row in rows],
            probability_clamped_flags=[row[8] for row in rows],
        )

```


---

## `src/lgd_bert/train.py`

```python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import DEFAULT_WANDB_PROJECT, ROOT, TrainConfig
from .data import load_glue_data, save_data_report
from .eval import evaluate
from .lsh import SimHashLSH
from .model_utils import (
    build_stored_vectors,
    classifier_head,
    classifier_query,
    compute_cls_representations,
    load_tokenizer_and_model,
    model_state_hash,
)
from .probability import corrected_loss_weights
from .samplers import LGDBatchSampler, RandomBatchSampler
from .wandb_utils import init_wandb, wandb_finish, wandb_log


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def make_run_dir(args) -> Path:
    output_root = Path(args.output_root) if args.output_root else ROOT / "runs"
    output_root.mkdir(parents=True, exist_ok=True)
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = args.run_name or f"{args.task}_{args.variant}_{args.correction}_seed{args.seed}_{stamp}"
        run_dir = output_root / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_csv_row(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_lsh_state(args, model, train_data, labels_np: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray, SimHashLSH, float]:
    start = time.time()
    cls_reps = compute_cls_representations(model, train_data, args.batch_size, device)
    stored_vectors = build_stored_vectors(
        cls_reps,
        labels_np,
        variant=args.variant,
        include_bias=args.include_bias_in_lsh,
    )
    lsh = SimHashLSH(
        dim=stored_vectors.shape[1],
        k=args.lsh_k,
        l=args.lsh_l,
        seed=args.seed + 17,
    )
    lsh.fit(stored_vectors)
    return cls_reps, stored_vectors, lsh, time.time() - start


def should_refresh(args, step: int, steps_per_epoch: int) -> bool:
    if args.variant == "random" or args.refresh_lsh in {"no_refresh", "initial_only"}:
        return False
    if args.refresh_lsh == "every_epoch":
        return step > 0 and steps_per_epoch > 0 and step % steps_per_epoch == 0
    if args.refresh_lsh == "every_n_steps":
        return step > 0 and step % args.refresh_steps == 0
    raise ValueError(f"Unsupported refresh_lsh mode: {args.refresh_lsh}")


def coverage_row(epoch_index: int, counts: Counter[int], train_size: int, steps_in_epoch: int, batch_size: int) -> dict:
    seen_counts = list(counts.values())
    unique_seen = len(seen_counts)
    repeated_sample_count = sum(1 for value in seen_counts if value > 1)
    duplicate_draw_count = sum(value - 1 for value in seen_counts if value > 1)
    never_seen_count = train_size - unique_seen
    return {
        "epoch": epoch_index,
        "train_size": train_size,
        "steps_in_epoch": steps_in_epoch,
        "batch_size": batch_size,
        "draw_count": sum(seen_counts),
        "unique_seen_count": unique_seen,
        "never_seen_count": never_seen_count,
        "never_seen_frac": never_seen_count / train_size,
        "repeated_sample_count": repeated_sample_count,
        "repeated_sample_frac": repeated_sample_count / train_size,
        "duplicate_draw_count": duplicate_draw_count,
        "duplicate_draw_frac_of_draws": duplicate_draw_count / max(sum(seen_counts), 1),
        "max_times_seen": max(seen_counts) if seen_counts else 0,
    }


def train_main(args) -> dict:
    set_seed(args.seed)
    os.environ.setdefault("HF_HOME", "/ssd_scratch/nilan/.cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/ssd_scratch/nilan/.cache/huggingface/datasets")
    run_dir = make_run_dir(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_tokenizer_and_model(args.model_name)
    model.to(device)
    if args.freeze_bert:
        base = model.bert if hasattr(model, "bert") else model.base_model
        for param in base.parameters():
            param.requires_grad = False

    weight, _bias = classifier_head(model)
    classifier_shape = list(weight.shape)
    data = load_glue_data(args.task, tokenizer, args.max_seq_length)
    save_data_report(data.report, run_dir / "data_report.json")
    labels_np = data.train.labels.numpy().astype("int64")
    steps_per_epoch = math.ceil(len(data.train) / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps is not None:
        total_steps = min(total_steps, int(args.max_steps))

    config = TrainConfig(**{key: getattr(args, key) for key in TrainConfig().__dict__ if hasattr(args, key)}).to_dict()
    config.update(
        {
            "run_dir": str(run_dir),
            "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_device_count": torch.cuda.device_count(),
            "torch_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "classifier_weight_shape": classifier_shape,
            "train_size_N": len(data.train),
            "validation_size": len(data.validation),
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "git_commit": git_commit(),
            "exact_command": " ".join(["python"] + os.sys.argv),
            "uniform_correction_check": "If p_i=1/N then 1/(N*p_i)=1 and corrected CE equals mean CE.",
        }
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    run = init_wandb(config, run_dir)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    if args.variant == "random":
        sampler = RandomBatchSampler(len(data.train), seed=args.seed + 101)
        stored_vectors = None
        lsh = None
        refresh_count = 0
        initial_refresh_time = 0.0
    else:
        _cls_reps, stored_vectors, lsh, initial_refresh_time = build_lsh_state(args, model, data.train, labels_np, device)
        sampler = LGDBatchSampler(lsh, stored_vectors, len(data.train), seed=args.seed + 101, probability_eps=args.probability_eps)
        refresh_count = 1
        summary = lsh.summary().__dict__
        (run_dir / "initial_lsh_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    start_time = time.time()
    last_eval = None
    status = "finished"
    error_message = None
    epoch_counts: Counter[int] = Counter()
    coverage_rows: list[dict] = []

    try:
        for step in range(1, total_steps + 1):
            step_start = time.time()
            refresh_time = 0.0
            if should_refresh(args, step - 1, steps_per_epoch):
                _cls_reps, stored_vectors, lsh, refresh_time = build_lsh_state(args, model, data.train, labels_np, device)
                sampler.update_index(lsh, stored_vectors)
                refresh_count += 1

            epoch = (step - 1) / max(steps_per_epoch, 1)
            if args.variant == "random":
                sample = sampler.sample_batch(args.batch_size)
                query_norm = 0.0
            else:
                query = classifier_query(model, args.variant, include_bias=args.include_bias_in_lsh)
                query_norm = float(np.linalg.norm(query))
                sample = sampler.sample_batch(args.batch_size, query)
            if args.audit_sample_coverage:
                epoch_counts.update(int(index) for index in sample.indices)

            batch = data.train.batch(sample.indices, device)
            labels = batch.pop("labels")
            outputs = model(**batch)
            logits = outputs.logits
            losses = F.cross_entropy(logits, labels, reduction="none")
            probabilities = torch.tensor(sample.probabilities, dtype=torch.float32, device=device)
            weight_stats = corrected_loss_weights(
                probabilities,
                train_size=len(data.train),
                mode=args.correction,
                max_weight=args.correction_max_weight,
                eps=args.probability_eps,
            )
            loss = (weight_stats.weights * losses).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}: {float(loss.detach().cpu())}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.max_grad_norm and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            wall = time.time() - start_time
            step_time = time.time() - step_start
            train_metrics = {
                "global_step": step,
                "epoch": epoch,
                "train/loss": float(loss.detach().cpu()),
                "time/wall_clock_sec": wall,
                "time/step_time_sec": step_time,
                "sampler/query_norm": query_norm,
                "correction/weight_mean": weight_stats.weight_mean,
                "correction/weight_min": weight_stats.weight_min,
                "correction/weight_max": weight_stats.weight_max,
                "correction/weight_clipped_frac": weight_stats.weight_clipped_frac,
                "lsh/refresh_count": refresh_count,
                "lsh/refresh_time_sec": refresh_time,
            }
            train_metrics.update(sample.stats())
            if step == 1 and initial_refresh_time:
                train_metrics["lsh/initial_refresh_time_sec"] = initial_refresh_time
            if step % args.log_every == 0:
                wandb_log(run, train_metrics)
                write_csv_row(run_dir / "train_log.csv", train_metrics)

            do_eval = step % args.eval_every == 0 or step == total_steps
            if do_eval:
                eval_metrics = evaluate(model, data.validation, args.batch_size, device)
                last_eval = {
                    "global_step": step,
                    "epoch": epoch,
                    **eval_metrics,
                    "paper_fig5/test_accuracy": eval_metrics["eval/accuracy"],
                    "paper_fig5/test_loss": eval_metrics["eval/loss"],
                    "time/wall_clock_sec": wall,
                }
                wandb_log(run, last_eval)
                write_csv_row(run_dir / "eval_log.csv", last_eval)
            if args.audit_sample_coverage and (step % steps_per_epoch == 0 or step == total_steps):
                epoch_index = len(coverage_rows)
                coverage = coverage_row(
                    epoch_index=epoch_index,
                    counts=epoch_counts,
                    train_size=len(data.train),
                    steps_in_epoch=steps_per_epoch,
                    batch_size=args.batch_size,
                )
                coverage_rows.append(coverage)
                write_csv_row(run_dir / "sample_coverage_by_epoch.csv", coverage)
                epoch_counts = Counter()
    except Exception as exc:
        status = "failed"
        error_message = repr(exc)
        raise
    finally:
        summary = {
            "status": status,
            "error_message": error_message,
            "run_dir": str(run_dir),
            "wandb_url": (run.url if run is not None else None),
            "final_eval": last_eval,
            "refresh_count": refresh_count,
            "model_state_hash": model_state_hash(model),
            "sample_coverage_by_epoch": coverage_rows,
        }
        (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        wandb_finish(run)
    return summary


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train BERT GLUE with random or LGD sampling.")
    parser.add_argument("--task", choices=["mrpc", "rte"], default="mrpc")
    parser.add_argument("--variant", choices=["random", "paper_lgd", "label_aware_lgd"], default="random")
    parser.add_argument("--correction", choices=["none", "full", "clipped", "sqrt", "normalized"], default="none")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--optimizer", choices=["adam"], default="adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--eval_every", type=int, default=25)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--lsh_k", type=int, default=7)
    parser.add_argument("--lsh_l", type=int, default=10)
    parser.add_argument("--refresh_lsh", choices=["initial_only", "every_epoch", "every_n_steps", "no_refresh"], default="every_epoch")
    parser.add_argument("--refresh_steps", type=int, default=100)
    parser.add_argument("--representation", choices=["cls_last_hidden_state"], default="cls_last_hidden_state")
    parser.add_argument("--include_bias_in_lsh", action="store_true")
    parser.add_argument("--correction_max_weight", type=float, default=10.0)
    parser.add_argument("--probability_eps", type=float, default=1e-12)
    parser.add_argument("--freeze_bert", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=0.0)
    parser.add_argument("--wandb_project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--audit_sample_coverage", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = train_main(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```


---

## `src/lgd_bert/wandb_utils.py`

```python
from __future__ import annotations

import os
from pathlib import Path


def init_wandb(config: dict, run_dir: Path):
    import wandb

    mode = config.get("wandb_mode", "online")
    if mode == "disabled":
        return None
    os.environ.setdefault("WANDB_PROJECT", config.get("wandb_project", "lgd_neurips2019_bert_repro"))
    os.environ.setdefault("WANDB_DIR", str(run_dir / "wandb"))
    run = wandb.init(
        project=config.get("wandb_project", "lgd_neurips2019_bert_repro"),
        group=config.get("wandb_group") or None,
        name=config.get("run_name") or None,
        mode=mode,
        dir=os.environ["WANDB_DIR"],
        config=config,
        tags=[
            config.get("task", "unknown"),
            config.get("variant", "unknown"),
            config.get("correction", "unknown"),
        ],
    )
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")
    if run.url:
        (run_dir / "wandb_url.txt").write_text(run.url + "\n")
        print(f"WANDB_URL={run.url}", flush=True)
    return run


def wandb_log(run, metrics: dict) -> None:
    if run is not None:
        run.log(metrics)


def wandb_finish(run) -> None:
    if run is not None:
        run.finish()

```


---

## `scripts/build_or_refresh_cache.py`

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lgd_bert.data import load_glue_data
from lgd_bert.model_utils import build_stored_vectors, compute_cls_representations, load_tokenizer_and_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a standalone CLS representation cache for inspection.")
    parser.add_argument("--task", choices=["mrpc", "rte"], required=True)
    parser.add_argument("--variant", choices=["paper_lgd", "label_aware_lgd"], default="label_aware_lgd")
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include_bias_in_lsh", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_tokenizer_and_model(args.model_name)
    model.to(device)
    data = load_glue_data(args.task, tokenizer, args.max_seq_length)
    cls = compute_cls_representations(model, data.train, args.batch_size, device)
    labels = data.train.labels.numpy().astype("int64")
    stored = build_stored_vectors(cls, labels, args.variant, include_bias=args.include_bias_in_lsh)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, cls=cls, stored=stored, labels=labels)
    meta = {
        "task": args.task,
        "variant": args.variant,
        "model_name": args.model_name,
        "representation": "cls_last_hidden_state",
        "shape_cls": list(cls.shape),
        "shape_stored": list(stored.shape),
        "data_report": data.report,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(out)


if __name__ == "__main__":
    main()

```


---

## `scripts/inspect_env.py`

```python
#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> dict:
    try:
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        return {"cmd": cmd, "returncode": result.returncode, "output": result.stdout}
    except FileNotFoundError as exc:
        return {"cmd": cmd, "returncode": 127, "output": repr(exc)}


def module_version(name: str) -> str | None:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception as exc:
        return "unavailable: " + repr(exc)


def main() -> None:
    out_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": stamp,
        "hostname": platform.node(),
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "versions": {
            "torch": module_version("torch"),
            "transformers": module_version("transformers"),
            "datasets": module_version("datasets"),
            "wandb": module_version("wandb"),
            "numpy": module_version("numpy"),
        },
        "commands": {
            "nvidia_smi": run(["nvidia-smi"]),
            "nvidia_smi_query": run(["nvidia-smi", "--query-gpu=index,name,memory.free,memory.total", "--format=csv,noheader"]),
            "free_h": run(["free", "-h"]),
            "df_h": run(["df", "-h", str(ROOT), "/ssd_scratch/nilan"]),
        },
        "disk_usage_root": shutil.disk_usage(ROOT)._asdict(),
    }
    try:
        import torch

        report["torch_cuda"] = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "cuda_version": torch.version.cuda,
        }
    except Exception as exc:
        report["torch_cuda"] = {"error": repr(exc)}
    json_path = out_dir / f"env_report_{stamp}.json"
    txt_path = out_dir / f"env_report_{stamp}.txt"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        f"timestamp: {stamp}",
        f"hostname: {report['hostname']}",
        f"python: {report['python']}",
        f"python_version: {report['python_version']}",
        f"cpu_count: {report['cpu_count']}",
        "",
        "versions:",
    ]
    lines.extend(f"  {k}: {v}" for k, v in report["versions"].items())
    lines.append("")
    for key, value in report["commands"].items():
        lines.append(f"## {key}")
        lines.append(value["output"])
    txt_path.write_text("\n".join(lines) + "\n")
    print(txt_path)
    print(json_path)


if __name__ == "__main__":
    main()

```


---

## `scripts/launch_sweep.py`

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def detect_gpus() -> list[int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,memory.total,name", "--format=csv,noheader,nounits"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return []
    rows = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        rows.append((int(parts[0]), int(parts[1]), int(parts[2]), parts[3]))
    rows.sort(key=lambda row: row[1], reverse=True)
    return [row[0] for row in rows]


def jobs_for_stage(stage: str, seeds: list[int]) -> list[dict]:
    if stage == "smoke":
        return [
            {"task": "mrpc", "variant": "random", "correction": "none", "seed": 0, "max_steps": 20},
            {"task": "mrpc", "variant": "label_aware_lgd", "correction": "none", "seed": 0, "max_steps": 20},
            {"task": "mrpc", "variant": "label_aware_lgd", "correction": "full", "seed": 0, "max_steps": 20},
            {"task": "rte", "variant": "random", "correction": "none", "seed": 0, "max_steps": 20},
            {"task": "rte", "variant": "label_aware_lgd", "correction": "none", "seed": 0, "max_steps": 20},
            {"task": "rte", "variant": "label_aware_lgd", "correction": "full", "seed": 0, "max_steps": 20},
        ]
    if stage == "sanity":
        return [
            {"task": task, "variant": variant, "correction": corr, "seed": 0, "max_steps": 100}
            for task in ["mrpc", "rte"]
            for variant, corr in [("random", "none"), ("paper_lgd", "none"), ("label_aware_lgd", "full")]
        ]
    if stage == "full_pair":
        return [
            {"task": "mrpc", "variant": "random", "correction": "none", "seed": 0, "max_steps": None},
            {"task": "mrpc", "variant": "label_aware_lgd", "correction": "full", "seed": 0, "max_steps": None},
        ]
    if stage == "coverage_audit":
        return [
            {"task": task, "variant": variant, "correction": "none", "seed": 0, "max_steps": None}
            for task in ["mrpc", "rte"]
            for variant in ["random", "paper_lgd", "label_aware_lgd"]
        ]
    if stage == "full_sweep":
        jobs = []
        for task in ["mrpc", "rte"]:
            for seed in seeds:
                jobs.append({"task": task, "variant": "random", "correction": "none", "seed": seed, "max_steps": None})
                for variant in ["paper_lgd", "label_aware_lgd"]:
                    for correction in ["none", "full"]:
                        jobs.append({"task": task, "variant": variant, "correction": correction, "seed": seed, "max_steps": None})
        return jobs
    raise ValueError(f"Unsupported stage: {stage}")


def command_for_job(job: dict, args) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_lgd_bert.py"),
        "--task",
        job["task"],
        "--variant",
        job["variant"],
        "--correction",
        job["correction"],
        "--seed",
        str(job["seed"]),
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--eval_every",
        str(args.eval_every),
        "--wandb_project",
        args.wandb_project,
        "--wandb_mode",
        args.wandb_mode,
        "--wandb_group",
        args.wandb_group,
        "--output_root",
        str(args.run_root),
        "--refresh_lsh",
        args.refresh_lsh,
    ]
    if job["max_steps"] is not None:
        cmd += ["--max_steps", str(job["max_steps"])]
    if args.correction_max_weight is not None:
        cmd += ["--correction_max_weight", str(args.correction_max_weight)]
    if args.audit_sample_coverage:
        cmd += ["--audit_sample_coverage"]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch queued independent BERT LGD runs across visible GPUs.")
    parser.add_argument("--stage", choices=["smoke", "sanity", "full_pair", "full_sweep", "coverage_audit"], default="smoke")
    parser.add_argument("--allow_full", action="store_true", help="Required for full_pair and full_sweep.")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--eval_every", type=int, default=25)
    parser.add_argument("--wandb_project", default="lgd_neurips2019_bert_repro")
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--refresh_lsh", choices=["initial_only", "every_epoch", "every_n_steps", "no_refresh"], default="every_epoch")
    parser.add_argument("--correction_max_weight", type=float, default=10.0)
    parser.add_argument("--audit_sample_coverage", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.stage.startswith("full") and not args.allow_full:
        raise SystemExit("Refusing full runs without --allow_full. Run smoke/sanity gates first.")
    if args.wandb_group is None:
        args.wandb_group = f"{args.stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.run_root = ROOT / "runs" / args.wandb_group
    args.run_root.mkdir(parents=True, exist_ok=True)

    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    jobs = jobs_for_stage(args.stage, seeds)
    gpus = detect_gpus()
    if not gpus:
        raise SystemExit("No GPUs detected by nvidia-smi.")
    command_log = args.run_root / "commands.txt"
    active: list[tuple[subprocess.Popen, int, object, object]] = []
    pending = list(jobs)
    with command_log.open("w") as log:
        for job in jobs:
            log.write(shlex.join(command_for_job(job, args)) + "\n")
    if args.dry_run:
        print(command_log)
        return

    while pending or active:
        active_gpus = {gpu for _proc, gpu, _stdout, _stderr in active}
        free_gpus = [gpu for gpu in gpus if gpu not in active_gpus]
        while pending and free_gpus:
            gpu = free_gpus.pop(0)
            job = pending.pop(0)
            name = f"{job['task']}_{job['variant']}_{job['correction']}_seed{job['seed']}_{int(time.time())}"
            stdout = (args.run_root / f"{name}.stdout.log").open("w")
            stderr = (args.run_root / f"{name}.stderr.log").open("w")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env.setdefault("HF_HOME", "/ssd_scratch/nilan/.cache/huggingface")
            env.setdefault("HF_DATASETS_CACHE", "/ssd_scratch/nilan/.cache/huggingface/datasets")
            env.setdefault("WANDB_DIR", str(args.run_root / "wandb"))
            cmd = command_for_job(job, args)
            print(f"START gpu={gpu} {shlex.join(cmd)}", flush=True)
            proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=stdout, stderr=stderr)
            active.append((proc, gpu, stdout, stderr))
        time.sleep(5)
        still_active = []
        for proc, gpu, stdout, stderr in active:
            code = proc.poll()
            if code is None:
                still_active.append((proc, gpu, stdout, stderr))
            else:
                stdout.close()
                stderr.close()
                print(f"FINISH gpu={gpu} returncode={code}", flush=True)
                if code != 0:
                    raise SystemExit(f"A run failed with return code {code}; see logs in {args.run_root}")
        active = still_active


if __name__ == "__main__":
    main()
```


---

## `scripts/run_lgd_bert.py`

```python
#!/usr/bin/env python3
from __future__ import annotations

from lgd_bert.train import main


if __name__ == "__main__":
    main()

```


---

## `scripts/run_sample_coverage_audit.py`

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer

from lgd_bert.data import load_glue_data
from lgd_bert.model_utils import classifier_query, load_tokenizer_and_model
from lgd_bert.samplers import LGDBatchSampler, RandomBatchSampler
from lgd_bert.train import build_lsh_state, coverage_row, set_seed


ROOT = Path(__file__).resolve().parents[1]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def audit_one(args, task: str, variant: str, output_root: Path) -> list[dict]:
    set_seed(args.seed)
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    data = load_glue_data(task, tokenizer, args.max_seq_length)
    train_size = len(data.train)
    steps_per_epoch = math.ceil(train_size / args.batch_size)
    total_steps = args.epochs * steps_per_epoch

    run_dir = output_root / f"{task}_{variant}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "task": task,
        "variant": variant,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "train_size": train_size,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "audit_mode": "sampler_only_static_initial_query_no_backprop",
        "note": "This audits sampler coverage over a full 3-epoch draw schedule. It does not update BERT/classifier weights.",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (run_dir / "data_report.json").write_text(json.dumps(data.report, indent=2, sort_keys=True) + "\n")

    if variant == "random":
        sampler = RandomBatchSampler(train_size, seed=args.seed + 101)
        query = None
    else:
        _tokenizer, model = load_tokenizer_and_model(args.model_name)
        model.to(device)
        labels_np = data.train.labels.numpy().astype("int64")
        lsh_args = SimpleNamespace(
            batch_size=args.batch_size,
            variant=variant,
            include_bias_in_lsh=False,
            lsh_k=args.lsh_k,
            lsh_l=args.lsh_l,
            seed=args.seed,
            probability_eps=args.probability_eps,
        )
        _cls_reps, stored_vectors, lsh, refresh_time = build_lsh_state(lsh_args, model, data.train, labels_np, device)
        sampler = LGDBatchSampler(lsh, stored_vectors, train_size, seed=args.seed + 101, probability_eps=args.probability_eps)
        query = classifier_query(model, variant, include_bias=False)
        (run_dir / "initial_lsh_summary.json").write_text(
            json.dumps({**lsh.summary().__dict__, "refresh_time_sec": refresh_time}, indent=2, sort_keys=True) + "\n"
        )

    rows: list[dict] = []
    counts: Counter[int] = Counter()
    fallback_count = 0
    draw_count = 0
    for step in range(1, total_steps + 1):
        if variant == "random":
            sample = sampler.sample_batch(args.batch_size)
        else:
            sample = sampler.sample_batch(args.batch_size, query)
        counts.update(int(index) for index in sample.indices)
        fallback_count += sum(1 for flag in sample.fallback_flags if flag)
        draw_count += len(sample.indices)
        if step % steps_per_epoch == 0 or step == total_steps:
            epoch_index = len(rows)
            row = coverage_row(epoch_index, counts, train_size, steps_per_epoch, args.batch_size)
            row.update(
                {
                    "task": task,
                    "variant": variant,
                    "seed": args.seed,
                    "fallback_draw_count": fallback_count,
                    "fallback_draw_frac": fallback_count / max(draw_count, 1),
                }
            )
            rows.append(row)
            counts = Counter()
            fallback_count = 0
            draw_count = 0
    write_csv(run_dir / "sample_coverage_by_epoch.csv", rows)
    return rows


def write_report(rows: list[dict], output_root: Path) -> None:
    report = output_root / "COVERAGE_AUDIT_REPORT.md"
    lines = [
        "# Sample Coverage Audit",
        "",
        "Audit mode: full 3-epoch sampler draw schedule, no BERT/classifier weight updates, no W&B.",
        "",
        "| Task | Variant | Epoch | Train N | Draws | Unique seen | Repeated samples | Duplicate draws | Never seen | Max times seen | Fallback frac |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {task} | {variant} | {epoch} | {train_size} | {draw_count} | {unique_seen_count} | "
            "{repeated_sample_count} | {duplicate_draw_count} | {never_seen_count} | {max_times_seen} | {fallback_draw_frac:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Definitions:",
            "",
            "- `Repeated samples`: number of distinct training examples sampled at least twice within that epoch.",
            "- `Duplicate draws`: total extra uses beyond the first use, i.e. `draw_count - unique_seen_count`.",
            "- `Never seen`: training examples with zero draws within that epoch.",
            "",
        ]
    )
    report.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit sampler replacement coverage over full epoch draw schedules.")
    parser.add_argument("--tasks", default="mrpc,rte")
    parser.add_argument("--variants", default="random,paper_lgd,label_aware_lgd")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--lsh_k", type=int, default=7)
    parser.add_argument("--lsh_l", type=int, default=10)
    parser.add_argument("--probability_eps", type=float, default=1e-12)
    parser.add_argument("--output_root", default=None)
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", "/home2/nilan/.cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/home2/nilan/.cache/huggingface/datasets")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else ROOT / "runs" / f"coverage_audit_sampler_only_{stamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for task in [item.strip() for item in args.tasks.split(",") if item.strip()]:
        for variant in [item.strip() for item in args.variants.split(",") if item.strip()]:
            rows = audit_one(args, task, variant, output_root)
            all_rows.extend(rows)
    write_csv(output_root / "sample_coverage_summary.csv", all_rows)
    write_report(all_rows, output_root)
    print(output_root)
    print(output_root / "sample_coverage_summary.csv")
    print(output_root / "COVERAGE_AUDIT_REPORT.md")


if __name__ == "__main__":
    main()

```


---

## `scripts/summarize_and_plot.py`

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def float_or_none(value):
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def label_for(config: dict) -> str:
    variant = config["variant"]
    correction = config["correction"]
    if variant == "random":
        return "SGD/random"
    if variant == "paper_lgd" and correction == "none":
        return "paper LGD"
    if variant == "paper_lgd":
        return f"paper LGD {correction}"
    if variant == "label_aware_lgd" and correction == "none":
        return "label-aware LGD"
    if variant == "label_aware_lgd" and correction == "full":
        return "label-aware LGD corrected"
    return f"{variant} {correction}"


def gather(run_root: Path) -> tuple[list[dict], list[dict]]:
    finals: list[dict] = []
    curves: list[dict] = []
    for config_path in sorted(run_root.glob("*/config.json")):
        run_dir = config_path.parent
        config = json.loads(config_path.read_text())
        summary_path = run_dir / "run_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        eval_rows = read_csv(run_dir / "eval_log.csv")
        train_rows = read_csv(run_dir / "train_log.csv")
        last_train = train_rows[-1] if train_rows else {}
        final_eval = summary.get("final_eval") or (eval_rows[-1] if eval_rows else {})
        base = {
            "run_dir": str(run_dir),
            "task": config.get("task"),
            "variant": config.get("variant"),
            "correction": config.get("correction"),
            "label": label_for(config),
            "seed": config.get("seed"),
            "wandb_url": summary.get("wandb_url"),
            "status": summary.get("status"),
            "refresh_count": summary.get("refresh_count"),
            "final_step": final_eval.get("global_step"),
            "final_accuracy": final_eval.get("eval/accuracy") or final_eval.get("paper_fig5/test_accuracy"),
            "final_loss": final_eval.get("eval/loss") or final_eval.get("paper_fig5/test_loss"),
            "final_f1": final_eval.get("eval/f1"),
            "wall_clock_sec": final_eval.get("time/wall_clock_sec"),
            "fallback_rate_last": last_train.get("sampler/fallback_rate"),
            "p_mean_last": last_train.get("sampler/p_mean"),
            "weight_mean_last": last_train.get("correction/weight_mean"),
            "weight_max_last": last_train.get("correction/weight_max"),
            "weight_clipped_frac_last": last_train.get("correction/weight_clipped_frac"),
        }
        finals.append(base)
        for row in eval_rows:
            curves.append(
                {
                    "task": config.get("task"),
                    "label": label_for(config),
                    "seed": config.get("seed"),
                    "global_step": int(float(row["global_step"])),
                    "accuracy": float(row["paper_fig5/test_accuracy"]),
                    "loss": float(row["paper_fig5/test_loss"]),
                }
            )
    return finals, curves


def write_table(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def mean_curves(curves: list[dict]) -> dict:
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in curves:
        grouped[(row["task"], row["label"], row["global_step"])].append(row)
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for (task, label, step), rows in grouped.items():
        out[(task, label)].append(
            {
                "global_step": step,
                "accuracy": sum(row["accuracy"] for row in rows) / len(rows),
                "loss": sum(row["loss"] for row in rows) / len(rows),
                "n": len(rows),
            }
        )
    for rows in out.values():
        rows.sort(key=lambda row: row["global_step"])
    return out


def plot(curves: list[dict], out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    averaged = mean_curves(curves)
    colors = {
        "SGD/random": "#1f77b4",
        "paper LGD": "#d62728",
        "paper LGD full": "#9467bd",
        "label-aware LGD": "#2ca02c",
        "label-aware LGD corrected": "#ff7f0e",
    }
    paths: list[Path] = []
    for task in ["mrpc", "rte"]:
        for metric, title_metric in [("accuracy", "Testing Accuracy"), ("loss", "Testing Loss")]:
            fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
            for (curve_task, label), rows in sorted(averaged.items()):
                if curve_task != task:
                    continue
                x = [row["global_step"] for row in rows]
                y = [row[metric] for row in rows]
                ax.plot(x, y, marker="o", linewidth=1.8, markersize=3, label=label, color=colors.get(label))
            ax.set_title(f"{task.upper()} {title_metric}")
            ax.set_xlabel("Iter")
            ax.set_ylabel("Accuracy" if metric == "accuracy" else "Loss")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            path = out_dir / f"{task}_{metric}_vs_iter.png"
            fig.savefig(path)
            plt.close(fig)
            paths.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), dpi=160)
    panel_specs = [
        ("mrpc", "accuracy", "MRPC Testing Accuracy"),
        ("rte", "accuracy", "RTE Testing Accuracy"),
        ("mrpc", "loss", "MRPC Testing Loss"),
        ("rte", "loss", "RTE Testing Loss"),
    ]
    for ax, (task, metric, title) in zip(axes.flat, panel_specs):
        for (curve_task, label), rows in sorted(averaged.items()):
            if curve_task != task:
                continue
            ax.plot(
                [row["global_step"] for row in rows],
                [row[metric] for row in rows],
                marker="o",
                linewidth=1.6,
                markersize=2.5,
                label=label,
                color=colors.get(label),
            )
        ax.set_title(title)
        ax.set_xlabel("Iter")
        ax.set_ylabel("Accuracy" if metric == "accuracy" else "Loss")
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    combined = out_dir / "figure5_bert_mrpc_rte_accuracy_loss.png"
    fig.savefig(combined)
    plt.close(fig)
    paths.append(combined)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize BERT LGD runs and export Figure-5-style plots.")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir) if args.out_dir else Path("plots") / run_root.name
    finals, curves = gather(run_root)
    write_table(finals, out_dir / "final_metrics.csv")
    write_table(curves, out_dir / "eval_curves.csv")
    paths = plot(curves, out_dir) if curves else []
    print(out_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

```


---

## `tests/test_corrected_loss.py`

```python
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

```


---

## `tests/test_label_aware_sign.py`

```python
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

```


---

## `tests/test_lsh_tables.py`

```python
from __future__ import annotations

import numpy as np

from lgd_bert.lsh import SimHashLSH


def test_every_index_appears_once_per_table():
    vectors = np.random.default_rng(0).normal(size=(32, 8)).astype("float32")
    lsh = SimHashLSH(dim=8, k=4, l=5, seed=0)
    lsh.fit(vectors)
    assert len(lsh.tables) == 5
    for table in lsh.tables:
        seen = sorted(index for bucket in table.values() for index in bucket)
        assert seen == list(range(32))
        assert all(0 <= key < 2**4 for key in table)

```


---

## `tests/test_probability_formula.py`

```python
from __future__ import annotations

import numpy as np

from lgd_bert.probability import lsh_sampling_probability, simhash_collision_probability


def test_collision_probability_monotonic_and_cp_k():
    low = simhash_collision_probability(-0.5)
    mid = simhash_collision_probability(0.0)
    high = simhash_collision_probability(0.8)
    assert low < mid < high
    vector = np.array([1.0, 0.0], dtype="float32")
    query = np.array([1.0, 0.0], dtype="float32")
    p, cp, cp_k, _cos, _clamped = lsh_sampling_probability(vector, query, k=7, bucket_size=4, attempts=2)
    assert np.isclose(cp_k, cp**7)
    assert np.isclose(p, cp_k * (1.0 - cp_k) / 4.0)

```


---

## `tests/test_sampler_shapes.py`

```python
from __future__ import annotations

import numpy as np

from lgd_bert.lsh import SimHashLSH
from lgd_bert.samplers import LGDBatchSampler


def test_sampler_returns_shapes_and_positive_probabilities():
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(40, 6)).astype("float32")
    lsh = SimHashLSH(dim=6, k=3, l=4, seed=2)
    lsh.fit(vectors)
    sampler = LGDBatchSampler(lsh, vectors, train_size=40, seed=3)
    batch = sampler.sample_batch(8, rng.normal(size=(6,)).astype("float32"))
    assert len(batch.indices) == 8
    assert len(batch.probabilities) == 8
    assert all(p > 0 for p in batch.probabilities)
    assert len(batch.bucket_sizes) == 8
    assert len(batch.attempts) == 8


def test_sampler_fallback_path():
    rng = np.random.default_rng(4)
    vectors = rng.normal(size=(10, 4)).astype("float32")
    lsh = SimHashLSH(dim=4, k=4, l=3, seed=5)
    lsh.fit(vectors)
    lsh.tables = [{} for _ in range(lsh.l)]
    sampler = LGDBatchSampler(lsh, vectors, train_size=10, seed=6)
    batch = sampler.sample_batch(5, rng.normal(size=(4,)).astype("float32"))
    assert all(batch.fallback_flags)
    assert all(np.isclose(p, 1.0 / 10) for p in batch.probabilities)

```

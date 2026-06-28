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

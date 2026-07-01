# Assumptions And Scope

## Immediate Problem

The earlier Sherlock audit found that the paper-like `K=7,L=10` LGD sampler collapsed badly at
`t=0` on MRPC. The failure signature was extremely poor coverage and heavy duplicate draws:

- MRPC LGD with replacement covered roughly `0.001` of train examples over 8192 draws.
- Batches had roughly `29/32` duplicate draws.
- Random coverage was much healthier.

This run treats sampling coverage and duplicate collapse as the first repair target.

## What Was Held Out Of Scope

The run deliberately did not attempt:

- RTE.
- probability correction.
- warmup variants.
- refresh-rate sweeps.
- large hyperparameter sweeps.
- smoke or sanity training runs.
- optimizer changes.

## Paper Anchors Used

The local paper files referenced by the run were:

- `/home2/nilan/research/AN/scratch_exps/papers/Experiments-NeurIPS-2019-fast-and-accurate-stochastic-gradient-estimation-Paper.pdf`
- `/home2/nilan/research/AN/scratch_exps/papers/experiments-lgd_supplement.pdf`

Anchors extracted into the notes:

- Main BERT experiment: MRPC/RTE, BERTbase, 3 epochs, batch size 32, Adam, `K=7,L=10`.
- `K` controls collision-probability decay and should be kept small enough to avoid empty or tiny
  bucket behavior.
- Logistic LGD motivates label-aware storage: store `y_i x_i`, query with `-theta`.
- BERT adaptation uses pooled representations in LSH tables, periodically refreshes tables, and uses
  classifier-layer parameters as the query.

The prompt also specified the supplement-style mini-batch collection rule: for a mini-batch, draw
multiple examples from a matching bucket when enough examples exist; otherwise continue sampling
from other matching buckets until the batch is filled. The implemented first-pass fix used
batch-without-replacement, not a full supplement-style bucket collector.

## Experiment Matrix

| condition | K | L | replacement | correction | refresh | warmup |
| --- | ---: | ---: | --- | --- | --- | --- |
| random epoch shuffle | n/a | n/a | epoch shuffle | none | n/a | none |
| label-aware LGD | 3 | 50 | batch without replacement | none | every epoch | none |
| label-aware LGD | 4 | 50 | batch without replacement | none | every epoch | none |
| label-aware LGD | 5 | 50 | batch without replacement | none | every epoch | none |
| label-aware LGD | 6 | 50 | batch without replacement | none | every epoch | none |

All runs used seed 0 and full MRPC 3-epoch fine-tuning.

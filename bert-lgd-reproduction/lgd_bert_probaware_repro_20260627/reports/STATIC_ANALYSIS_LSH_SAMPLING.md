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


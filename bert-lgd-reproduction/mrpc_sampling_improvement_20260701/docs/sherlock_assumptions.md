# Sherlock Audit Assumptions

Created for `lgd_bert_sherlock_audit_20260628`.

## Paper Details Verified Directly

- Main paper PDF pages were rendered to `reports/pdf_renders/main_paper-*.png`.
- Supplement PDF pages were rendered to `reports/pdf_renders/supplement-*.png`.
- The supplement states that BERT sequence classification uses the final hidden state for the first token as the fixed-dimensional pooled representation.
- The supplement states that pooled representations are preprocessed in LSH tables and classifier-layer parameters are used as queries.
- The supplement says representations can be periodically updated, but does not specify the refresh frequency.
- The main paper gives BERT settings: MRPC/RTE, BERTbase, batch size 32, 3 epochs, Adam, K=7, L=10.

## Ambiguities Kept Explicit

- The exact BERT checkpoint name is not specified; this reproduction keeps `bert-base-uncased`.
- The learning-rate exponent remains ambiguous in the supplied PDF render; this reproduction keeps `2e-5`.
- The paper does not specify Adam betas, epsilon, weight decay, schedule, warmup, max sequence length, or evaluation cadence.
- The paper does not spell out binary-softmax query reduction for BERT. The implementation uses `w1 - w0`.
- The label-aware BERT variant is an extension inspired by the logistic-regression derivation, where labels are in `{-1,+1}`, data is stored as `y_i x_i`, and the query is `-theta`.
- The exact importance probability under no-replacement batch or epoch modes is not derived here; those correction modes are marked heuristic/biased.

## Implementation Facts To Verify With Audits

- GLUE label IDs must be printed per task before assuming label `1` is the positive class.
- `K=7` means `2^7 = 128` possible SimHash buckets per table, not 64.
- Current random baseline used sampling with replacement. A standard shuffled DataLoader-style baseline is a separate `--random_mode epoch_shuffle` condition.
- Existing completed runs did not log per-sample indices, so duplicate rates from those finished runs cannot be reconstructed exactly.

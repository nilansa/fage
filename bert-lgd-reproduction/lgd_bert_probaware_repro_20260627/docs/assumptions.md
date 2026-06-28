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


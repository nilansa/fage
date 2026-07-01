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

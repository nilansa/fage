# Paper Notes

- Main PDF: `/home2/nilan/research/AN/scratch_exps/papers/Experiments-NeurIPS-2019-fast-and-accurate-stochastic-gradient-estimation-Paper.pdf`.
- Main PDF anchors used here: BERTbase on MRPC/RTE; 3 epochs; batch size 32; Adam; LSH K=7, L=10; K controls collision-probability decay; logistic LGD stores label-multiplied inputs and queries with negative classifier direction.
- Supplement PDF: `/home2/nilan/research/AN/scratch_exps/papers/experiments-lgd_supplement.pdf`.
- Supplement PDF anchors used here: BERT pooled representations are stored in LSH tables, tables can be periodically refreshed, and classifier-layer parameters are used as sampling queries.
- The supplement-style mini-batch bucket-filling rule was taken from the run prompt; it was not separately visible in the extracted two-page local supplement text.

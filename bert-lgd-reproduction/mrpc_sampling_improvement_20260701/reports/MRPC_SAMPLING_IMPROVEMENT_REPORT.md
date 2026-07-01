# MRPC Sampling Improvement Report

## Folder

Created folder: `/home2/nilan/research/AN/scratch_exps/bert_lgd_sherlock_sampling_improvement_20260701_1525`

Copied source folder: `/home2/nilan/research/AN/scratch_exps/fage_sherlock_tmp/bert-lgd-reproduction/lgd_bert_sherlock_audit_20260628`

## Paper Files

- `/home2/nilan/research/AN/scratch_exps/papers/Experiments-NeurIPS-2019-fast-and-accurate-stochastic-gradient-estimation-Paper.pdf` (found)
- `/home2/nilan/research/AN/scratch_exps/papers/experiments-lgd_supplement.pdf` (found)

See `paper_notes.md` for the paper anchors used in this run.

## Commands And GPU Assignment

- GPU 0: `/ssd_scratch/nilan/venvs/lgd-bert-sherlock-20260628/bin/python /home2/nilan/research/AN/scratch_exps/bert_lgd_sherlock_sampling_improvement_20260701_1525/code/lgd_bert_sherlock_audit_20260628/scripts/run_lgd_bert.py --task mrpc --variant random --correction none --seed 0 --run_name MRPC_random_epoch_shuffle_seed0 --run_dir /home2/nilan/research/AN/scratch_exps/bert_lgd_sherlock_sampling_improvement_20260701_1525/runs/MRPC_random_epoch_shuffle_seed0 --batch_size 32 --epochs 3 --lr 2e-05 --optimizer adam --eval_every 25 --wandb_project lgd_bert_sampling_improvement_mrpc --wandb_mode online --wandb_group sampling_improvement_mrpc --lsh_k 0 --lsh_l 0 --refresh_lsh every_epoch --replacement_mode with_replacement --random_mode epoch_shuffle --sampler_start_step 0 --sampler_warmup none --correction_max_weight 10.0 --audit_sample_coverage --log_sample_indices`
- GPU 1: `/ssd_scratch/nilan/venvs/lgd-bert-sherlock-20260628/bin/python /home2/nilan/research/AN/scratch_exps/bert_lgd_sherlock_sampling_improvement_20260701_1525/code/lgd_bert_sherlock_audit_20260628/scripts/run_lgd_bert.py --task mrpc --variant label_aware_lgd --correction none --seed 0 --run_name MRPC_LA_K3_L50_BWoR_refreshEpoch_noCorr_seed0 --run_dir /home2/nilan/research/AN/scratch_exps/bert_lgd_sherlock_sampling_improvement_20260701_1525/runs/MRPC_LA_K3_L50_BWoR_refreshEpoch_noCorr_seed0 --batch_size 32 --epochs 3 --lr 2e-05 --optimizer adam --eval_every 25 --wandb_project lgd_bert_sampling_improvement_mrpc --wandb_mode online --wandb_group sampling_improvement_mrpc --lsh_k 3 --lsh_l 50 --refresh_lsh every_epoch --replacement_mode batch_without_replacement --random_mode epoch_shuffle --sampler_start_step 0 --sampler_warmup none --correction_max_weight 10.0 --audit_sample_coverage --log_sample_indices`
- GPU 2: `/ssd_scratch/nilan/venvs/lgd-bert-sherlock-20260628/bin/python /home2/nilan/research/AN/scratch_exps/bert_lgd_sherlock_sampling_improvement_20260701_1525/code/lgd_bert_sherlock_audit_20260628/scripts/run_lgd_bert.py --task mrpc --variant label_aware_lgd --correction none --seed 0 --run_name MRPC_LA_K4_L50_BWoR_refreshEpoch_noCorr_seed0 --run_dir /home2/nilan/research/AN/scratch_exps/bert_lgd_sherlock_sampling_improvement_20260701_1525/runs/MRPC_LA_K4_L50_BWoR_refreshEpoch_noCorr_seed0 --batch_size 32 --epochs 3 --lr 2e-05 --optimizer adam --eval_every 25 --wandb_project lgd_bert_sampling_improvement_mrpc --wandb_mode online --wandb_group sampling_improvement_mrpc --lsh_k 4 --lsh_l 50 --refresh_lsh every_epoch --replacement_mode batch_without_replacement --random_mode epoch_shuffle --sampler_start_step 0 --sampler_warmup none --correction_max_weight 10.0 --audit_sample_coverage --log_sample_indices`
- GPU 3: `/ssd_scratch/nilan/venvs/lgd-bert-sherlock-20260628/bin/python /home2/nilan/research/AN/scratch_exps/bert_lgd_sherlock_sampling_improvement_20260701_1525/code/lgd_bert_sherlock_audit_20260628/scripts/run_lgd_bert.py --task mrpc --variant label_aware_lgd --correction none --seed 0 --run_name MRPC_LA_K5_L50_BWoR_refreshEpoch_noCorr_seed0 --run_dir /home2/nilan/research/AN/scratch_exps/bert_lgd_sherlock_sampling_improvement_20260701_1525/runs/MRPC_LA_K5_L50_BWoR_refreshEpoch_noCorr_seed0 --batch_size 32 --epochs 3 --lr 2e-05 --optimizer adam --eval_every 25 --wandb_project lgd_bert_sampling_improvement_mrpc --wandb_mode online --wandb_group sampling_improvement_mrpc --lsh_k 5 --lsh_l 50 --refresh_lsh every_epoch --replacement_mode batch_without_replacement --random_mode epoch_shuffle --sampler_start_step 0 --sampler_warmup none --correction_max_weight 10.0 --audit_sample_coverage --log_sample_indices`
- GPU 0: `/ssd_scratch/nilan/venvs/lgd-bert-sherlock-20260628/bin/python /home2/nilan/research/AN/scratch_exps/bert_lgd_sherlock_sampling_improvement_20260701_1525/code/lgd_bert_sherlock_audit_20260628/scripts/run_lgd_bert.py --task mrpc --variant label_aware_lgd --correction none --seed 0 --run_name MRPC_LA_K6_L50_BWoR_refreshEpoch_noCorr_seed0 --run_dir /home2/nilan/research/AN/scratch_exps/bert_lgd_sherlock_sampling_improvement_20260701_1525/runs/MRPC_LA_K6_L50_BWoR_refreshEpoch_noCorr_seed0 --batch_size 32 --epochs 3 --lr 2e-05 --optimizer adam --eval_every 25 --wandb_project lgd_bert_sampling_improvement_mrpc --wandb_mode online --wandb_group sampling_improvement_mrpc --lsh_k 6 --lsh_l 50 --refresh_lsh every_epoch --replacement_mode batch_without_replacement --random_mode epoch_shuffle --sampler_start_step 0 --sampler_warmup none --correction_max_weight 10.0 --audit_sample_coverage --log_sample_indices`

## W&B

Project: `lgd_bert_sampling_improvement_mrpc`

Mode: `online`

Runs / offline paths:

- `MRPC_random_epoch_shuffle_seed0`: https://wandb.ai/nilansa/lgd_bert_sampling_improvement_mrpc/runs/7ppykeie
- `MRPC_LA_K3_L50_BWoR_refreshEpoch_noCorr_seed0`: https://wandb.ai/nilansa/lgd_bert_sampling_improvement_mrpc/runs/qngb9dsj
- `MRPC_LA_K4_L50_BWoR_refreshEpoch_noCorr_seed0`: https://wandb.ai/nilansa/lgd_bert_sampling_improvement_mrpc/runs/0j21c5js
- `MRPC_LA_K5_L50_BWoR_refreshEpoch_noCorr_seed0`: https://wandb.ai/nilansa/lgd_bert_sampling_improvement_mrpc/runs/3qf9upw9
- `MRPC_LA_K6_L50_BWoR_refreshEpoch_noCorr_seed0`: https://wandb.ai/nilansa/lgd_bert_sampling_improvement_mrpc/runs/u18xrcc8

## Final MRPC Metrics

| run_name | K | L | replacement_mode | correction | refresh_mode | final_eval_accuracy | final_eval_f1 | final_eval_loss | best_eval_accuracy | best_eval_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MRPC_random_epoch_shuffle_seed0 |  |  | epoch_shuffle | none | every_epoch | 0.845588 | 0.892308 | 0.503034 | 0.862745 | 0.902778 |
| MRPC_LA_K3_L50_BWoR_refreshEpoch_noCorr_seed0 | 3 | 50 | batch_without_replacement | none | every_epoch | 0.82598 | 0.87344 | 0.452323 | 0.828431 | 0.882943 |
| MRPC_LA_K4_L50_BWoR_refreshEpoch_noCorr_seed0 | 4 | 50 | batch_without_replacement | none | every_epoch | 0.794118 | 0.854671 | 0.52446 | 0.794118 | 0.856176 |
| MRPC_LA_K5_L50_BWoR_refreshEpoch_noCorr_seed0 | 5 | 50 | batch_without_replacement | none | every_epoch | 0.781863 | 0.83964 | 0.521427 | 0.781863 | 0.853896 |
| MRPC_LA_K6_L50_BWoR_refreshEpoch_noCorr_seed0 | 6 | 50 | batch_without_replacement | none | every_epoch | 0.821078 | 0.880914 | 0.561492 | 0.821078 | 0.880914 |


## Sampling Health

| run_name | K | coverage_after_training | mean_unique_samples_per_batch | mean_duplicate_samples_per_batch | mean_fallback_rate | mean_bucket_size | sampled_label1_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MRPC_random_epoch_shuffle_seed0 |  | 1 | 31.8957 | 0 | 0 | 3668 | 0.674482 |
| MRPC_LA_K3_L50_BWoR_refreshEpoch_noCorr_seed0 | 3 | 0.91494 | 32 | 0 | 0 | 547.091 | 0.582609 |
| MRPC_LA_K4_L50_BWoR_refreshEpoch_noCorr_seed0 | 4 | 0.852781 | 32 | 0 | 0 | 236.755 | 0.580888 |
| MRPC_LA_K5_L50_BWoR_refreshEpoch_noCorr_seed0 | 5 | 0.703381 | 32 | 0 | 0 | 97.6997 | 0.494203 |
| MRPC_LA_K6_L50_BWoR_refreshEpoch_noCorr_seed0 | 6 | 0.566249 | 32 | 0 | 0 | 72.0997 | 0.517935 |


## Verdict

- Sampling improved versus the collapsed K=7,L=10 reference: `True`.
- Best K by sampling health: `3`.
- Best K by MRPC accuracy/F1: `3`.
- Any LGD variant beat random epoch-shuffle: `False`.
- Decision: Coverage is no longer the main failure; next check should be query-loss correlation / whether label-aware score actually selects high-loss MRPC examples.

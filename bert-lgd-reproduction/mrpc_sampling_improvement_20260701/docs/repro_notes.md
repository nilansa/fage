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

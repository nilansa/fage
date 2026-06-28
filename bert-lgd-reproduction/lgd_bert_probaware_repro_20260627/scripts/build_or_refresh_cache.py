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


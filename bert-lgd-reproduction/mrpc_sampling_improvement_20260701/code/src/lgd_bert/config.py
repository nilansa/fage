from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER_MAIN = Path("/home2/nilan/research/AN/scratch_exps/papers/Experiments-NeurIPS-2019-fast-and-accurate-stochastic-gradient-estimation-Paper.pdf")
PAPER_SUPPLEMENT = Path("/home2/nilan/research/AN/scratch_exps/papers/experiments-lgd_supplement.pdf")
DEFAULT_WANDB_PROJECT = "lgd_bert_sherlock_audit"
DEFAULT_MODEL = "bert-base-uncased"


@dataclass
class TrainConfig:
    task: str = "mrpc"
    variant: str = "random"
    correction: str = "none"
    seed: int = 0
    model_name: str = DEFAULT_MODEL
    batch_size: int = 32
    epochs: int = 3
    lr: float = 2e-5
    optimizer: str = "adam"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    max_seq_length: int = 128
    eval_every: int = 25
    log_every: int = 1
    max_steps: int | None = None
    lsh_k: int = 7
    lsh_l: int = 10
    hash_family: str = "simhash_signed_random_projection"
    refresh_lsh: str = "every_epoch"
    refresh_steps: int = 100
    replacement_mode: str = "with_replacement"
    random_mode: str = "with_replacement"
    sampler_start_step: int = 0
    sampler_warmup: str = "none"
    log_sample_indices: bool = False
    representation: str = "cls_last_hidden_state"
    include_bias_in_lsh: bool = False
    correction_max_weight: float = 10.0
    probability_eps: float = 1e-12
    freeze_bert: bool = False
    max_grad_norm: float = 0.0
    wandb_project: str = DEFAULT_WANDB_PROJECT
    wandb_mode: str = "online"
    wandb_group: str | None = None
    run_name: str | None = None
    run_dir: str | None = None
    output_root: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["paper_main"] = str(PAPER_MAIN)
        data["paper_supplement"] = str(PAPER_SUPPLEMENT)
        return data

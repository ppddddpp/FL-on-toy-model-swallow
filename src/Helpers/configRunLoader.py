from dataclasses import dataclass, field
from typing import List
import yaml

@dataclass
class ConfigRun:
    experiment_case: str = "SYBIL_ONLY"

    attacker_ids: List[str] = field(default_factory=list)

    num_rounds: int = 5
    local_epochs: int = 5
    seed: int = 2709
    enable_prover: bool = True

    # Sybil Configuration
    sybil_mode: str = "static"          # static | leader | coordinated
    alpha: float = 0.8
    sybil_collusion: bool = True
    sybil_fake_data_size: int = 400
    sybil_history_window: int = 5
    sybil_use_grad: str = "NONE"        
    
    sybil_amplification_factor: float = 1.0

    @staticmethod
    def load(path: str) -> "ConfigRun":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        return ConfigRun(
            experiment_case = raw.get("experiment_case", "SYBIL_ONLY"),
            attacker_ids    = raw.get("attacker_ids", []),

            num_rounds      = raw.get("num_rounds", 5),
            local_epochs    = raw.get("local_epochs", 5),
            seed            = raw.get("seed", 2709),
            enable_prover   = raw.get("enable_prover", True),

            sybil_mode                 = raw.get("sybil_mode", "static"),
            alpha                      = raw.get("alpha", 0.8),
            sybil_collusion            = raw.get("sybil_collusion", True),
            sybil_fake_data_size       = raw.get("sybil_fake_data_size", 400),
            sybil_history_window       = raw.get("sybil_history_window", 5),
            sybil_use_grad             = raw.get("sybil_use_grad", "NONE"),
            
            sybil_amplification_factor = raw.get("sybil_amplification_factor", 1.0),
        )
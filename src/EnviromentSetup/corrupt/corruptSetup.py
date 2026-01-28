from pathlib import Path
from typing import Any, Dict

from Helpers.configRunLoader import ConfigRun
from EnviromentSetup.corrupt.participationAttack.sybilAmplificationAttack import (
    SybilAmplificationAttack,
)
from EnviromentSetup.corrupt.prover.participationAttackOnSybilProving import (
    MaliciousContributionsOnSybilProving,
)

VALID_EXPERIMENTS = {"SYBIL_ONLY"}

class ExperimentConfig:
    """
    Simplified Config Wrapper for Sybil-Only Experiments.
    """
    def __init__(self, run_cfg: Any):
        BASE_DIR = Path(__file__).resolve().parents[3]
        self.run_cfg = run_cfg if run_cfg is not None else ConfigRun.load(BASE_DIR / "config" / "config_run.yaml")
        self.experiment_case = getattr(run_cfg, "experiment_case", "SYBIL_ONLY")

        if self.experiment_case not in VALID_EXPERIMENTS:
            raise ValueError(f"Unknown experiment_case='{self.experiment_case}'. Expected 'SYBIL_ONLY'")

        self.sybil_attack_type = self.run_cfg.sybil_mode

    def summary(self) -> str:
        return f"experiment_case={self.experiment_case} (Sybil Detection Mode)"


class AttackEngines:
    BASE_PROOF_DIR = Path("logs/proofs").resolve()

    def __init__(self, run_cfg, base_proof_dir=BASE_PROOF_DIR):
        BASE_DIR = Path(__file__).resolve().parents[3]
        
        # Load Config
        self.run_cfg = ConfigRun.load(BASE_DIR / "config" / "config_run.yaml") if run_cfg is None else run_cfg
        self.enable_prover = self.run_cfg.enable_prover
        self.exp = ExperimentConfig(self.run_cfg)
        self.attacker_ids = set(self.run_cfg.attacker_ids)
        self.BASE_PROOF_DIR = base_proof_dir

        self.sybil_engine = SybilAmplificationAttack(
            amplification_factor=self.run_cfg.sybil_amplification_factor,
            sybil_mode=self.run_cfg.sybil_mode,
            shared_vector=None,
            fake_data_size=self.run_cfg.sybil_fake_data_size,
            alpha=self.run_cfg.alpha,
            collusion=self.run_cfg.sybil_collusion
        )

        self.sybil_prover_cls = (
            MaliciousContributionsOnSybilProving
            if self.enable_prover else None
        )
        
    def make_sybil_prover(self, round_id):
        """
        Creates a prover instance to audit Sybil attacks.
        """
        if self.sybil_prover_cls is None:
            return None
        return self.sybil_prover_cls(
            probe_name=f"SYBIL_r{round_id}",
            output_dir=self.BASE_PROOF_DIR
        )
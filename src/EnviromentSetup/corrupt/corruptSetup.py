import torch
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Any, Tuple, Dict, List

from Helpers.configRunLoader import ConfigRun
from EnviromentSetup.corrupt.maliciousContributions.maliciousContributionsOnData import (
    MaliciousContributionsGeneratorOnData,
)
from EnviromentSetup.corrupt.maliciousContributions.maliciousContributionsOnGradient import (
    MaliciousContributionsGeneratorOnGradient,
)
from EnviromentSetup.corrupt.participationAttack.freeRiderAttack import (
    FreeRiderAttack,
    FreeRiderDataAttack,
)
from EnviromentSetup.corrupt.participationAttack.sybilAmplificationAttack import (
    SybilAmplificationAttack,
)

from EnviromentSetup.corrupt.prover.maliciousContributionsOnDataProving import (
    MaliciousContributionsGeneratorOnDataProving,
)

from EnviromentSetup.corrupt.prover.maliciousContributionsOnGradientProving import (
    MaliciousContributionsOnGradientProving,
)

from EnviromentSetup.corrupt.prover.participationAttackOnFreeRiderProving import (
    MaliciousContributionsOnFreeRiderProving,
)

from EnviromentSetup.corrupt.prover.participationAttackOnSybilProving import (
    MaliciousContributionsOnSybilProving,
)

from Helpers.Helpers import (
    log_and_print,
    toy_dataset_to_df, 
    df_to_toy_dataset
)

VALID_EXPERIMENTS = {
    "MC_DATA",
    "MC_GRAD",
    "MC_BOTH",
    "FR_DATA",
    "FR_GRAD",
    "FR_BOTH",
    "SYBIL_ONLY",
}


class ExperimentConfig:
    def __init__(self, run_cfg: Any):
        self.run_cfg = run_cfg if run_cfg is not None else ConfigRun.load(Path(__file__).resolve().parents[3] / "config" / "config_run.yaml")
        self.experiment_case = getattr(run_cfg, "experiment_case", None)
        if self.experiment_case is None:
            atk_cat = getattr(run_cfg, "attack_category", None)
            atk_mode = getattr(run_cfg, "attack_mode", None)
            if atk_cat == "maliciousContributions":
                if atk_mode == "data":
                    self.experiment_case = "MC_DATA"
                elif atk_mode == "grad":
                    self.experiment_case = "MC_GRAD"
                elif atk_mode == "both":
                    self.experiment_case = "MC_BOTH"
            elif atk_cat == "participationAttack":
                if atk_mode == "data":
                    self.experiment_case = "FR_DATA"
                elif atk_mode == "grad":
                    self.experiment_case = "FR_GRAD"
                elif atk_mode == "both":
                    self.experiment_case = "FR_BOTH"
            else:
                # default
                self.experiment_case = "MC_DATA"

        if self.experiment_case not in VALID_EXPERIMENTS:
            raise ValueError(f"Unknown experiment_case='{self.experiment_case}'. Choose one of {VALID_EXPERIMENTS}")

        # Derived booleans
        self.is_mc = self.experiment_case.startswith("MC")
        self.is_fr = self.experiment_case.startswith("FR")
        self.is_sybil_only = self.experiment_case == "SYBIL_ONLY"
        self.data_mode = "_DATA" in self.experiment_case or "_BOTH" in self.experiment_case
        self.grad_mode = (
            "_GRAD" in self.experiment_case
            or "_BOTH" in self.experiment_case
        )

        self.mc_attack_type = self.run_cfg.mc_data_attack_type
        self.mc_grad_attack_type = self.run_cfg.mc_grad_attack_type
        self.fr_attack_type = self.run_cfg.free_rider_mode
        self.sybil_attack_type = self.run_cfg.sybil_mode

    def summary(self) -> str:
        return (f"experiment_case={self.experiment_case} is_mc={self.is_mc} is_fr={self.is_fr} "
                f"is_sybil_only={self.is_sybil_only} data_mode={self.data_mode} grad_mode={self.grad_mode}")


class AttackEngines:
    BASE_PROOF_DIR = Path("logs/proofs").resolve()

    def __init__(self, run_cfg, base_proof_dir=BASE_PROOF_DIR):
        # load configs
        BASE_DIR = Path(__file__).resolve().parents[3]
        run_cfg = ConfigRun.load(BASE_DIR / "config" / "config_run.yaml") if run_cfg is None else run_cfg
        
        self.run_cfg = run_cfg
        self.enable_prover = run_cfg.enable_prover
        
        # --- Config & Attacker Set ---
        self.exp = ExperimentConfig(run_cfg)
        self.attacker_ids = set(run_cfg.attacker_ids)

        # data engines
        self.mc_data_template = MaliciousContributionsGeneratorOnData(
            df=pd.DataFrame(),
            text_col=run_cfg.text_col,
            label_col=run_cfg.label_col,
            client_col=run_cfg.client_col,
            seed=run_cfg.seed,
            suspect_fraction=run_cfg.suspect_fraction
        )

        self.mc_data_prover_cls = (
            MaliciousContributionsGeneratorOnDataProving
            if self.enable_prover else None
        )

        # gradient engines
        self.mc_grad_engine = MaliciousContributionsGeneratorOnGradient(
            attack_type=run_cfg.mc_grad_attack_type,
            scale_factor=run_cfg.mc_grad_scale_factor,
            seed=run_cfg.seed,
            history_window=run_cfg.mc_grad_window
        )
        
        self.mc_grad_prover_cls = (
            MaliciousContributionsOnGradientProving
            if self.enable_prover else None
        )

        self.free_rider_grad_engine = FreeRiderAttack(
            mode=run_cfg.free_rider_grad_mode,
            fake_data_size=run_cfg.fake_data_size_data,
            noise_scale=run_cfg.noise_scale,
            seed=run_cfg.seed
        )
        
        self.free_rider_data_engine = FreeRiderDataAttack(
            mode=run_cfg.free_rider_mode,
            tiny_fraction=run_cfg.tiny_fraction,
            duplicate_factor=run_cfg.duplicate_factor,
            fake_data_size=run_cfg.fake_data_size_data,
            random_noise_dim=run_cfg.random_noise_dim,
            seed=run_cfg.seed
        )

        self.free_rider_prover_cls = (
            MaliciousContributionsOnFreeRiderProving
            if self.enable_prover else None
        )

        self.sybil_engine = SybilAmplificationAttack(
            amplification_factor=run_cfg.sybil_amplification_factor,
            sybil_mode=run_cfg.sybil_mode,
            shared_vector=None,                       # updated later each round
            fake_data_size=run_cfg.sybil_fake_data_size,
            alpha=run_cfg.alpha,
            collusion=run_cfg.sybil_collusion
        )

        self.sybil_prover_cls = (
            MaliciousContributionsOnSybilProving
            if self.enable_prover else None
        )

    # ---------- MC DATA ----------
    def make_mc_data_prover(self, original_df, corrupted_df, client_id, round_id):
        if self.mc_data_prover_cls is None:
            return None
        return self.mc_data_prover_cls(
            original_df=original_df,
            corrupted_df=corrupted_df,
            text_col=self.run_cfg.text_col,
            label_col=self.run_cfg.label_col,
            probe_name=f"MC_DATA_{client_id}_r{round_id}",
            output_dir=self.BASE_PROOF_DIR
        )


    # ---------- MC GRAD ----------
    def make_mc_grad_prover(self, clean_grad, attacked_grad, client_id, round_id):
        if self.mc_grad_prover_cls is None:
            return None
        return self.mc_grad_prover_cls(
            clean_grad=clean_grad,
            attacked_grad=attacked_grad,
            probe_name=f"MC_GRAD_{client_id}_r{round_id}",
            output_dir=self.BASE_PROOF_DIR
        )

    # ---------- FREE RIDER ----------
    def make_free_rider_prover(self, client_id, round_id):
        if self.free_rider_prover_cls is None:
            return None
        return self.free_rider_prover_cls(
            probe_name=f"FR_{client_id}_r{round_id}",
            output_dir=self.BASE_PROOF_DIR
        )


    # ---------- SYBIL ----------
    def make_sybil_prover(self, round_id):
        if self.sybil_prover_cls is None:
            return None
        return self.sybil_prover_cls(
            probe_name=f"SYBIL_r{round_id}",
            output_dir=self.BASE_PROOF_DIR
        )

    def apply_data_attacks(self, client_dataset, client_id: str, round_id: int, label2id=None, log_file=None) -> Tuple[Any, Any]:
        """
        Apply data-level attacks (MC-DATA or FR-DATA) if applicable.
        Returns: (attacked_dataset, fake_num_samples)
        """
        fake_num_samples = None
        ds = client_dataset
        exp = self.exp

        # ---------- MC-DATA ----------
        if exp.is_mc and exp.data_mode and client_id in self.attacker_ids:
            log_and_print(f"[MC-DATA] Attacking client {client_id}", log_file=log_file)
            try:
                # convert to DataFrame
                src_df = getattr(ds, "df", None)
                if isinstance(src_df, pd.DataFrame):
                    df = src_df.copy()
                else:
                    df = toy_dataset_to_df(ds)
                    df = df.rename(columns={"label": "Group", "text": "Information"})

                # prepare instance
                mc_cls = self.mc_data_template.__class__
                mc_instance = mc_cls(
                    df=df.copy(),
                    text_col=self.run_cfg.text_col,
                    label_col=self.run_cfg.label_col,
                    client_col=self.run_cfg.client_col,
                    seed=self.run_cfg.seed,
                    suspect_fraction=self.run_cfg.suspect_fraction,
                    log_file=log_file,
                    label2id=label2id,
                )

                # execute attack
                if exp.mc_attack_type == "random_label_flip": mc_instance.random_label_flip()
                elif exp.mc_attack_type == "random_text_noise": mc_instance.random_text_noise()
                elif exp.mc_attack_type == "semantic_noise": mc_instance.semantic_noise()
                elif exp.mc_attack_type == "backdoor": mc_instance.add_backdoor_trigger()
                elif exp.mc_attack_type == "duplicate_flood": mc_instance.duplicate_flood()
                elif exp.mc_attack_type == "ood": mc_instance.ood_injection()
                elif exp.mc_attack_type == "targeted_flip":
                    mc_instance.targeted_label_flip(src_label=self.run_cfg.src_label, tgt_label=self.run_cfg.tgt_label)
                else:
                    raise ValueError(f"Unknown mc_attack_type={exp.mc_attack_type}")

                corrupted_df = mc_instance.get_corrupted_dataset()

                # prover
                if self.enable_prover:
                    prover = self.make_mc_data_prover(df, corrupted_df, client_id, round_id)
                    if prover:
                        try:
                            prover.run()
                        except Exception as e:
                            log_and_print(f"[Prover DEBUG] run() raised: {e}", log_file=log_file)

                ds = df_to_toy_dataset(corrupted_df, client_dataset)
                fake_num_samples = len(corrupted_df)
                log_and_print(f"[ATTACK][MC-DATA] {client_id} applied ({exp.mc_attack_type})", log_file=log_file)
            except Exception as e:
                log_and_print(f"[WARN] MC-DATA {client_id}: {e}", log_file=log_file)

        # ---------- FREE-RIDER DATA ----------
        if exp.is_fr and exp.data_mode and client_id in self.attacker_ids:
            log_and_print(f"[FR-DATA] Attacking client {client_id}", log_file=log_file)
            try:
                src_df = getattr(ds, "df", None)
                if isinstance(src_df, pd.DataFrame):
                    df_for_fr = src_df.copy()
                else:
                    df_for_fr = toy_dataset_to_df(ds)
                    df_for_fr = df_for_fr.rename(columns={"text": self.run_cfg.text_col, "label": self.run_cfg.label_col})

                fr_out = self.free_rider_data_engine.apply(df_for_fr, metadata={"num_samples": len(df_for_fr)})
                
                if isinstance(fr_out, tuple) and len(fr_out) == 2:
                    modified_df, meta = fr_out
                else:
                    modified_df, meta = fr_out, {}

                ds = df_to_toy_dataset(modified_df, client_dataset)
                fake_num_samples = meta.get("num_samples", fake_num_samples)
                log_and_print(f"[ATTACK][FR-DATA] {client_id} mode={self.free_rider_data_engine.mode}", log_file=log_file)

                if self.enable_prover:
                    prover = self.make_free_rider_prover(client_id, round_id)
                    if prover:
                        try:
                            prover.run(dataset=modified_df, metadata={"num_samples": fake_num_samples, "original_num_samples": len(df_for_fr)})
                        except Exception as e:
                            log_and_print(f"[Prover DEBUG][FR-DATA] run() raised: {e}", log_file=log_file)

            except Exception as e:
                log_and_print(f"[WARN] FR-DATA {client_id}: {e}", log_file=log_file)

        return ds, fake_num_samples

    def apply_grad_attacks(self, delta_np: Dict[str, np.ndarray], client_id: str, 
                           num_samples: int, prev_updates: list, round_id: int, log_file=None) -> Tuple[Dict[str, np.ndarray], int]:
        """
        Apply gradient-level attacks (MC-GRAD or FR-GRAD).
        """
        # Deep copy to ensure safety
        d_np = {k: np.array(v, copy=True) for k, v in delta_np.items()}
        # Clean copy for history/provers
        clean_copy = {k: np.array(v, copy=True) for k, v in delta_np.items()}
        exp = self.exp

        def _validate_and_convert(out_dict, src_keys):
            if out_dict is None: raise ValueError("attack returned None")
            converted = {}
            for k in src_keys:
                arr = np.array(out_dict[k], copy=True)
                if arr.shape != d_np[k].shape:
                    raise ValueError(f"Shape mismatch key {k}")
                converted[k] = arr
            return converted

        # --- MC-Grad ---
        if exp.is_mc and exp.grad_mode and client_id in self.attacker_ids and self.run_cfg.mc_grad_delta:
            try:
                log_and_print(f"[MC-GRAD] Attacking client {client_id}", log_file=log_file)
                if prev_updates is not None:
                    attacked = self.mc_grad_engine.generate(d_np, prev_updates=prev_updates)
                else:
                    attacked = self.mc_grad_engine.generate(d_np)
                
                d_np = _validate_and_convert(attacked, d_np.keys())

                if self.enable_prover:
                    prover = self.make_mc_grad_prover(clean_copy, d_np, client_id, round_id)
                    if prover: 
                        try: prover.run()
                        except Exception as e: log_and_print(f"[Prover] Error: {e}", log_file=log_file)
                
                log_and_print(f"[ATTACK][MC-GRAD-DELTA] client={client_id}", log_file=log_file)
            except Exception as e:
                log_and_print(f"[WARN] MC-GRAD-DELTA client={client_id} failed: {e}", log_file=log_file)

        # --- FreeRider Grad ---
        if exp.is_fr and exp.grad_mode and client_id in self.attacker_ids:
            try:
                log_and_print(f"[FR-GRAD] Attacking client {client_id}", log_file=log_file)
                gr, meta = self.free_rider_grad_engine.apply(d_np, client_metadata={"num_samples": num_samples})
                d_np = _validate_and_convert(gr, d_np.keys())

                if self.enable_prover:
                    prover = self.make_free_rider_prover(client_id, round_id)
                    if prover:
                        history_hashes = [prover._grad_hash(u) for u in prev_updates]
                        try:
                            prover.run(benign_update=clean_copy, malicious_update=d_np, metadata={"num_samples": num_samples}, history_hashes=history_hashes)
                        except Exception as e: log_and_print(f"[Prover] Error: {e}", log_file=log_file)

                num_samples = meta.get("num_samples", num_samples)
                log_and_print(f"[ATTACK][FR-GRAD] client={client_id}", log_file=log_file)
            except Exception as e:
                log_and_print(f"[WARN] FR-GRAD client={client_id} failed: {e}", log_file=log_file)

        return d_np, num_samples
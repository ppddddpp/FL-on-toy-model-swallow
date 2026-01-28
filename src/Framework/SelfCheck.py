from pathlib import Path
import numpy as np
from typing import Dict, Any

from .sybil_check import SybilClustering
from Helpers.Helpers import log_and_print

BASE_DIR = Path(__file__).resolve().parents[3]

class SelfCheckManager:
    """
    Single-Layer Defense: Identity Verification (Sybil Detection).
    """
    def __init__(
            self, 
            global_model=None, 
            log_dir=None, 
            **kwargs
        ):
        self.log_dir = log_dir if log_dir else BASE_DIR / "logs" / "run.txt"

        self.sybil_check = SybilClustering(threshold=0.98) 
        self.global_model = global_model

    def run_round(self, client_updates: Dict[str, Any], round_id=1, global_model=None, **kwargs):
        log_and_print(f"\n[SelfCheck] Round {round_id} started with {len(client_updates)} clients.", log_file=self.log_dir)

        kept_ids = self.sybil_check.filter_sybils(client_updates)
        
        # Calculate who was dropped
        all_ids = set(client_updates.keys())
        kept_set = set(kept_ids)
        dropped_sybils = all_ids - kept_set
        
        if dropped_sybils:
            log_and_print(f"[Stage 1] Dropped {len(dropped_sybils)} Sybils: {dropped_sybils}", log_file=self.log_dir)

        decisions = {}
        scores = {}

        for cid in client_updates:
            if cid in kept_set:
                decisions[cid] = "ACCEPT"
                scores[cid] = 1.0  # Trust Score = 100%
            else:
                decisions[cid] = "REJECT"
                scores[cid] = 0.0  # Trust Score = 0%

        # Logging
        counts = {
            "accepted": list(decisions.values()).count("ACCEPT"), 
            "rejected": list(decisions.values()).count("REJECT")
        }
        
        log_and_print(f"[Round {round_id}] Final: {counts['accepted']} Accepted, {counts['rejected']} Rejected.", log_file=self.log_dir)

        # Return standard format expected by Server
        return decisions, scores, {"decisions": decisions, "counts": counts, "trust_scores": scores}
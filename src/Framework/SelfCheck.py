from pathlib import Path
import numpy as np
import torch
from typing import Dict, Any

from .sybil_check import SybilClustering
from Helpers.Helpers import log_and_print

BASE_DIR = Path(__file__).resolve().parents[3]

class SelfCheckManager:
    """
    Single-Layer Defense: Identity Verification (Sybil Detection).
    Includes a 'Hard Trap' for identical updates to catch cold-start attacks.
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

        suspicious_clones = set()
        
        # 1. Flatten updates to single vectors for comparison
        client_ids = list(client_updates.keys())
        flat_vectors = []
        
        # Ensure consistent ordering of keys when flattening
        if client_ids:
            sample_keys = sorted(client_updates[client_ids[0]].keys())
            
            for cid in client_ids:
                tensors = [client_updates[cid][k].float().view(-1) for k in sample_keys]
                flat_vectors.append(torch.cat(tensors))

        if len(flat_vectors) > 1:
            # Stack into a matrix (Num_Clients x Total_Params)
            update_stack = torch.stack(flat_vectors)
            
            # Normalize vectors to calculate Cosine Similarity efficiently
            # CosSim(A, B) = (A . B) / (|A| * |B|)
            norm = update_stack.norm(p=2, dim=1, keepdim=True)
            normalized_stack = update_stack / (norm + 1e-8)
            
            # Matrix Multiply: Result is a Client x Client similarity matrix
            sim_matrix = torch.mm(normalized_stack, normalized_stack.t())
            
            # Check for clones (Similarity > 0.999)
            for i in range(len(client_ids)):
                for j in range(i + 1, len(client_ids)):
                    similarity = sim_matrix[i, j].item()
                    
                    if similarity > 0.95:
                        suspicious_clones.add(client_ids[j])

        if suspicious_clones:
            log_and_print(f"[SelfCheck] [TRAP] Round {round_id} detected {len(suspicious_clones)} clones: {suspicious_clones}", log_file=self.log_dir)

        kept_ids_list = self.sybil_check.filter_sybils(client_updates)
        kept_set = set(kept_ids_list)
        kept_set = kept_set - suspicious_clones
        
        # Calculate who was dropped
        all_ids = set(client_updates.keys())
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
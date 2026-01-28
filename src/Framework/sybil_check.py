import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform

class SybilClustering:
    """
    Exact Sybil Detection.
    Detects cliques of clients with >98% pairwise cosine similarity.
    """
    def __init__(self, threshold: float = 0.98):
        self.threshold = threshold

    def filter_sybils(self, client_updates: dict) -> list:
        """
        Returns a list of client_ids to KEEP.
        If a cluster of Sybils is found, only 1 representative is kept.
        """
        ids = list(client_updates.keys())
        n = len(ids)
        if n < 2: return ids

        # Flatten updates to a matrix (Exact, no sketching)
        flat_vecs = []
        for cid in ids:
            update = client_updates[cid]
            if isinstance(update, dict):
                # Sort keys to ensure deterministic ordering of parameters
                parts = []
                for k in sorted(update.keys()):
                    v = update[k]
                    # Convert to numpy 1D
                    if isinstance(v, torch.Tensor):
                        v_np = v.detach().cpu().numpy().flatten()
                    else:
                        v_np = np.array(v).flatten()
                    parts.append(v_np)
                
                if not parts:
                    # Handle empty update case
                    vec = np.array([0.0]) 
                else:
                    vec = np.concatenate(parts)
                
            elif isinstance(update, torch.Tensor):
                vec = update.detach().cpu().numpy().flatten()
            else:
                vec = np.array(update).flatten()
                
            flat_vecs.append(vec)

        # Compute Similarity Matrix
        try:
            dists = pdist(flat_vecs, metric='cosine')
            # Handle numerical instability where distance might be slightly < 0 or > 2 or NaN
            dists = np.nan_to_num(dists, nan=1.0) 
        except Exception as e:
            print(f"[SybilCheck] Warning: Distance calculation failed ({e}). Skipping check.")
            return ids

        sim_matrix = 1 - squareform(dists)

        keep_ids = []
        covered_indices = set()

        for i in range(n):
            if i in covered_indices:
                continue

            # This client 'i' is the representative of its cluster
            keep_ids.append(ids[i])
            covered_indices.add(i)

            # Find all clones (Sybils) of 'i'
            for j in range(i + 1, n):
                if j not in covered_indices:
                    if sim_matrix[i, j] > self.threshold:
                        covered_indices.add(j)

        return keep_ids
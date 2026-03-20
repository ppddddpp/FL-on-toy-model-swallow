import numpy as np
import torch

class SybilClustering:
    def __init__(self, threshold: float = 0.98):
        self.threshold = threshold

    def filter_sybils(self, client_updates: dict) -> list:
        ids = list(client_updates.keys())
        n = len(ids)
        if n < 2:
            return ids

        flat_vecs = []
        for cid in ids:
            update = client_updates[cid]

            if isinstance(update, dict):
                parts = []
                for k in sorted(update.keys()):
                    v = update[k]
                    if isinstance(v, torch.Tensor):
                        v_np = v.detach().cpu().numpy().flatten()
                    else:
                        v_np = np.array(v).flatten()
                    parts.append(v_np)

                vec = np.concatenate(parts) if parts else np.array([0.0])

            elif isinstance(update, torch.Tensor):
                vec = update.detach().cpu().numpy().flatten()
            else:
                vec = np.array(update).flatten()

            flat_vecs.append(vec)

        X = np.vstack(flat_vecs)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        # Cosine similarity
        sim_matrix = X @ X.T
        sim_matrix = np.nan_to_num(sim_matrix, nan=0.0)

        # Directional dominance (secondary)
        mean_vec = np.median(X, axis=0)
        norm_mean = np.linalg.norm(mean_vec) + 1e-8

        alignments = [
            np.dot(vec, mean_vec) / ((np.linalg.norm(vec) + 1e-8) * norm_mean)
            for vec in X
        ]

        dominant_indices = [i for i, a in enumerate(alignments) if a > 0.9]
        use_dominance = len(dominant_indices) > max(3, n * 0.6)

        # Clustering
        keep_ids = []
        covered_indices = set()

        for i in range(n):
            if i in covered_indices:
                continue

            keep_ids.append(ids[i])
            covered_indices.add(i)

            for j in range(i + 1, n):
                if j in covered_indices:
                    continue

                if sim_matrix[i, j] > self.threshold or (
                    use_dominance and
                    i in dominant_indices and
                    j in dominant_indices and
                    sim_matrix[i, j] > 0.9
                ):
                    covered_indices.add(j)

        return keep_ids
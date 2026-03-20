import numpy as np
import torch

class SybilClustering:
    def __init__(self, threshold: float = 0.98, min_cluster_size: int = 3):
        self.threshold = threshold
        self.min_cluster_size = min_cluster_size

    def filter_sybils(self, client_updates: dict) -> list:
        ids = list(client_updates.keys())
        n = len(ids)
        if n < 2:
            return ids

        # Flatten updates
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

        # Normalize
        X = np.vstack(flat_vecs)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        # Cosine similarity
        sim_matrix = X @ X.T
        sim_matrix = np.nan_to_num(sim_matrix, nan=0.0)

        # Build adjacency graph
        adj = sim_matrix > self.threshold

        # Find connected components
        visited = set()
        clusters = []

        for i in range(n):
            if i in visited:
                continue

            stack = [i]
            component = []

            while stack:
                node = stack.pop()
                if node in visited:
                    continue

                visited.add(node)
                component.append(node)

                neighbors = np.where(adj[node])[0]
                for nb in neighbors:
                    if nb not in visited:
                        stack.append(nb)

            clusters.append(component)

        # Decide Sybil vs Honest
        keep_ids = []

        for cluster in clusters:
            if len(cluster) >= self.min_cluster_size:
                # Check tightness
                submatrix = sim_matrix[np.ix_(cluster, cluster)]
                min_sim = np.min(submatrix)

                if min_sim > (self.threshold - 0.01):  # e.g. 0.97
                    # Sybil cluster -> keep 1
                    keep_ids.append(ids[cluster[0]])
                    continue

            # Honest cluster -> keep all
            for idx in cluster:
                keep_ids.append(ids[idx])

        return keep_ids
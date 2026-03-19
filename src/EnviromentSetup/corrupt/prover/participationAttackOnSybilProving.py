import numpy as np
from Helpers.Helpers import ensure_dir, save_json
from Helpers.Helpers import log_and_print
from sklearn.cluster import DBSCAN

class MaliciousContributionsOnSybilProving:
    """
    Detects Sybil amplification attacks in Federated Learning.
    Works at:
        - single-client malicious update
        - multi-client similarity clustering
        - metadata consistency checks
        - amplification detection
        - collusion / shared vector detection
    """

    def __init__(self, probe_name="sybil_proof", output_dir="logs/proofs", eps=0.05):
        """
        benign_update : dict
            The real gradient for reference.

        malicious_updates : dict
            key = client_id
            value = gradient dict

        client_metadatas : dict
            key = client_id
            value = metadata containing num_samples, etc.
        """
        self.probe_name = probe_name
        self.output_dir = output_dir

        self.eps = eps
        self.benign = None
        self.updates = {}
        self.metadatas = {}
        self.fake_datasize = 50000  # threshold for fake data size detection
        self._observed = False
        self.log_file = self.output_dir / f"{self.probe_name}_log.txt"

    def observe(self, benign_update, malicious_updates, client_metadatas, fake_data_size=None):
        """
        Collect all updates for this round.
        """
        self.benign = benign_update
        self.updates = malicious_updates
        self.metadatas = client_metadatas
        self._observed = True

        if fake_data_size is not None:
            self.fake_datasize = fake_data_size

    def _flatten(self, grad):
        return np.concatenate([v.flatten() for v in grad.values()])

    def detect_amplification(self, grad):
        g = self._flatten(grad)

        # Case 1: benign reference exists (MC/FR + Sybil)
        if self.benign is not None:
            b = self._flatten(self.benign)
            return float(np.linalg.norm(g) / (np.linalg.norm(b) + 1e-8))

        # Case 2: SYBIL_ONLY -> no benign reference
        # Use population-relative amplification
        norms = [
            np.linalg.norm(self._flatten(u))
            for u in self.updates.values()
        ]

        mean_norm = np.mean(norms) + 1e-8
        return float(np.linalg.norm(g) / mean_norm)
    
    def detect_density_clusters(self):
        client_ids = list(self.updates.keys())
        if len(client_ids) == 0:
            return {}

        X = np.vstack([self._flatten(self.updates[cid]) for cid in client_ids])

        clustering = DBSCAN(eps=self.eps, min_samples=2, metric='cosine').fit(X)
        labels = clustering.labels_

        clusters = {}
        for cid, label in zip(client_ids, labels):
            if label == -1:
                continue

            label = int(label)

            clusters.setdefault(label, []).append(cid)

        return clusters

    def detect_directional_alignment(self):
        client_ids = list(self.updates.keys())
        X = np.vstack([self._flatten(self.updates[cid]) for cid in client_ids])

        mean_vec = np.mean(X, axis=0)
        norm_mean = np.linalg.norm(mean_vec) + 1e-8

        alignment_scores = {}
        for cid, vec in zip(client_ids, X):
            alignment_scores[cid] = float(
                np.dot(vec, mean_vec) /
                ((np.linalg.norm(vec) + 1e-8) * norm_mean)
            )

        return alignment_scores

    def detect_shared_pattern(self):
        """
        Detects if many clients have exactly the same gradient direction,
        a strong sign of Sybil collusion.
        """

        flattened = [self._flatten(g) for g in self.updates.values()]
        arr = np.vstack(flattened)

        # Standard deviation across clients at every parameter location
        per_param_std = np.std(arr, axis=0)

        # if std is extremely small then we have identical vectors
        if np.mean(per_param_std) < 1e-6:
            return True, float(np.mean(per_param_std))

        return False, float(np.mean(per_param_std))

    def detect_fake_data_size(self):
        inconsistencies = []
        for cid, meta in self.metadatas.items():
            if meta.get("num_samples", 0) > self.fake_datasize:
                inconsistencies.append(cid)
        return inconsistencies

    def summary(self):
        amp_ratios = {
            cid: self.detect_amplification(grad)
            for cid, grad in self.updates.items()
        }

        density_clusters = self.detect_density_clusters()
        shared_pattern, shared_std = self.detect_shared_pattern()
        fake_size_clients = self.detect_fake_data_size()
        alignment_scores = self.detect_directional_alignment()

        attack_types = []

        aligned_clients = sum(1 for v in alignment_scores.values() if v > 0.9)
        alignment_ratio = aligned_clients / (len(alignment_scores) + 1e-8)

        if alignment_ratio > 0.5:
            attack_types.append("Directional Sybil Alignment")

        if any(r > 3.0 for r in amp_ratios.values()):
            attack_types.append("Amplification Attack")
        
        cluster_sizes = {k: len(v) for k, v in density_clusters.items()}
        num_clustered_clients = sum(cluster_sizes.values())
        total_clients = len(self.updates)
        cluster_ratio = num_clustered_clients / (total_clients + 1e-8)

        if any(len(c) > 2 for c in density_clusters.values()):
            attack_types.append("Sybil Collusion (Density-Based)")

        if shared_pattern:
            attack_types.append("Sybil Shared Vector Attack")

        if fake_size_clients:
            attack_types.append("Fake Data Size Manipulation")

        if not attack_types:
            attack_types = ["No detectable Sybil attack"]

        return {
            "amplification_ratios": amp_ratios,
            "alignment_scores": alignment_scores,
            "alignment_ratio": alignment_ratio,
            "cluster_ratio": cluster_ratio,
            "cluster_sizes": cluster_sizes,
            "density_clusters": density_clusters,
            "shared_vector_detected": shared_pattern,
            "shared_vector_std": shared_std,
            "fake_data_size_clients": fake_size_clients,
            "detected_attack_types": attack_types,
        }

    def run(self):
        if not self._observed:
            raise RuntimeError("SybilProver.run() called before observe()")

        if not self.updates:
            msg = "status : skipped | reason : no_sybil_updates"
            log_and_print(msg, log_file=self.log_file)
            return 

        ensure_dir(self.output_dir)

        report = self.summary()

        save_json(report, self.output_dir / "sybil_report.json")
        return report
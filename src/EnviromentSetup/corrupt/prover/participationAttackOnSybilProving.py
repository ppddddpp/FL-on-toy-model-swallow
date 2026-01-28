import numpy as np
from scipy.spatial.distance import cosine
from collections import defaultdict
from Helpers.Helpers import ensure_dir, save_json
from Helpers.Helpers import log_and_print

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

    def __init__(self, probe_name="sybil_proof", output_dir="logs/proofs"):
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

    def detect_collusion(self):
        """
        Returns dictionary:
            { client_id: {other_client: cosine_similarity} }
        and detects clusters of nearly identical gradients.
        """
        client_ids = list(self.updates.keys())
        vectors = {cid: self._flatten(self.updates[cid]) for cid in client_ids}

        similarities = defaultdict(dict)
        collusion_groups = []

        for i in range(len(client_ids)):
            for j in range(i + 1, len(client_ids)):
                c1, c2 = client_ids[i], client_ids[j]
                v1, v2 = vectors[c1], vectors[c2]

                sim = 1 - cosine(v1, v2)  # cosine similarity
                similarities[c1][c2] = sim
                similarities[c2][c1] = sim

        # Detect colluding pairs/groups (cosine similarity very close to 1)
        threshold = 0.995
        for c1 in client_ids:
            group = {c1}
            for c2 in client_ids:
                if c1 != c2 and similarities[c1].get(c2, 0) > threshold:
                    group.add(c2)
            if len(group) > 1:
                collusion_groups.append(group)

        return similarities, collusion_groups

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

        similarities, collusion_groups = self.detect_collusion()
        shared_pattern, shared_std = self.detect_shared_pattern()
        fake_size_clients = self.detect_fake_data_size()

        attack_types = []

        if any(r > 3.0 for r in amp_ratios.values()):
            attack_types.append("Amplification Attack")

        if collusion_groups:
            attack_types.append("Sybil Collusion (Similarity)")

        if shared_pattern:
            attack_types.append("Sybil Shared Vector Attack")

        if fake_size_clients:
            attack_types.append("Fake Data Size Manipulation")

        if not attack_types:
            attack_types = ["No detectable Sybil attack"]

        return {
            "amplification_ratios": amp_ratios,
            "collusion_similarity_matrix": similarities,
            "collusion_groups": collusion_groups,
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
        if "collusion_groups" in report:
            report["collusion_groups"] = [list(group) for group in report["collusion_groups"]]

        save_json(report, self.output_dir / "sybil_report.json")
        return report
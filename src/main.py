import json
import csv
import torch
import numpy as np
import random
from pathlib import Path
import datetime
from typing import Dict, List

from Server.server import Server
from Client.client import Client
from DataHandler.dataset_builder import DatasetBuilder
from Helpers.configLoader import Config
from Helpers.configRunLoader import ConfigRun
from Helpers.safe_ops import safe_param_subtract
from EnviromentSetup.model.model import ToyBERTClassifier
from EnviromentSetup.corrupt.corruptSetup import ExperimentConfig, AttackEngines
from Helpers.Helpers import _device_from_state_dict, numpy_delta_to_torch, torch_delta_to_numpy
from Helpers.Helpers import log_and_print

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    cfg = Config.load(BASE_DIR / "config" / "config.yaml")
    run_cfg = ConfigRun.load(BASE_DIR / "config" / "config_run.yaml")
    attacker_ids = set(run_cfg.attacker_ids)

    # Logging setup
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_file = log_dir / "run.txt"
    proof_dir = BASE_DIR / "logs" / "proofs"
    proof_dir.mkdir(parents=True, exist_ok=True)

    log_and_print(f"Attacker IDs: {attacker_ids}", log_file=run_file)

    # Set seeds
    random.seed(run_cfg.seed)
    np.random.seed(run_cfg.seed)
    torch.manual_seed(run_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_cfg.seed)

    # Build experiment config & validate
    exp = ExperimentConfig(run_cfg)
    log_and_print(f"[ConfigRun] {exp.summary()} attackers={attacker_ids}", log_file=run_file)

    # Build attack engines
    engines = AttackEngines(run_cfg, base_proof_dir=proof_dir)

    server = Server(
        model_cls=ToyBERTClassifier,
        config=cfg,
        device="cuda" if torch.cuda.is_available() else "cpu",
        text_col="Information",
        label_col="Group",
        checkpoint_dir="checkpoints/base_model",
        log_dir=run_file
    )

    client_paths = [
        BASE_DIR / "data" / "animal" / f"n{i}" / f"client_{i}_data.csv"
        for i in range(1, 13)
    ]

    _, _, _, vocab_base, label2id_base = DatasetBuilder.build_dataset(
        path=BASE_DIR / "data" / "animal" / "base" / "base_model.csv",
        max_len=cfg.max_seq_len,
        text_col="Information",
        label_col="Group"
    )

    clients = []
    for i, path in enumerate(client_paths):
        log_and_print(f"[ClientSetup] Loading client {i+1}", log_file=run_file)

        train_ds, val_ds, test_ds, vocab, label2id = DatasetBuilder.build_dataset(
            path=path,
            max_len=cfg.max_seq_len,
            vocab=vocab_base,
            label2id=label2id_base.copy(),
            text_col="Information",
            label_col="Group"
        )

        vocab_size = train_ds.vocab_size
        num_classes = len(label2id)

        def make_model_fn(vs=vocab_size, nc=num_classes, c=cfg):
            return lambda: ToyBERTClassifier(
                vocab_size=vs, num_classes=nc,
                d_model=c.model_dim, nhead=c.num_heads,
                num_layers=c.num_layers, dim_ff=c.ffn_dim,
                max_len=c.max_seq_len, dropout=c.dropout
            )
        
        # Create client object
        client_obj = Client(
            client_id=f"client_{i+1}",
            model_fn=make_model_fn(),
            dataset=train_ds,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        # Note: No mc_grad_engine attachment needed for Sybil Only

        clients.append({
            "id": f"client_{i+1}",
            "label2id": label2id,
            "client": client_obj,
            "val": val_ds,
            "test": test_ds
        })

    # Global test set for evaluation
    _, _, global_test_ds, _, _ = DatasetBuilder.build_dataset(
        path=BASE_DIR / "data" / "animal" / "base" / "base_model.csv",
        max_len=cfg.max_seq_len,
        text_col="Information",
        label_col="Group"
    )

    # Log setup
    json_path = log_dir / "accuracy_log.json"
    csv_path = log_dir / "accuracy_log.csv"
    if not json_path.exists():
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)

    def write_csv_header(num_clients):
        cols = ["round", "timestamp", "global_acc"]
        for i in range(1, num_clients + 1):
            cols += [f"client_{i}_acc", f"client_{i}_samples"]
        cols += ["rejected_clients"] # Add column for tracking rejections
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(cols)

    write_csv_header(len(clients))

    global_weights = server.global_model.state_dict()
    num_rounds = run_cfg.num_rounds
    local_epochs = run_cfg.local_epochs
    previous_sybil_updates: List[List[Dict[str, np.ndarray]]] = []

    for rnd in range(1, num_rounds + 1):
        log_and_print(f"\n[Main] Round {rnd}/{num_rounds} - exp={exp.experiment_case}", log_file=run_file)
        
        client_updates = []
        per_client_metrics = []
        current_sybil_updates_np = [] # Store raw updates for Sybil Engine

        for cb in clients:
            client_obj = cb["client"]
            client_id = cb["id"]

            # Train
            new_weights, num_samples, discovered = client_obj.local_train(
                global_weights=global_weights,
                epochs=local_epochs,
                batch_size=cfg.batch_size,
                lr=cfg.lr
            )

            # Eval
            client_acc = client_obj.evaluate(weights=new_weights, batch_size=cfg.batch_size)
            log_and_print(f"[LocalEval] {client_id} acc={client_acc:.4f}", log_file=run_file)

            # Compute Delta
            device = _device_from_state_dict(global_weights)
            new_weights = {k: v.to(device) for k, v in new_weights.items()}
            delta = {}
            for k in global_weights.keys():
                try:
                    delta[k] = safe_param_subtract(new_weights[k], global_weights[k])
                except:
                    delta[k] = torch.zeros_like(global_weights[k])

            # Prepare for Sybil Engine (if attacker)
            if client_id in attacker_ids:
                # Convert to numpy for the attack engine
                delta_np = torch_delta_to_numpy(delta)
                current_sybil_updates_np.append((client_id, delta_np))

            client_updates.append({
                "client_id": client_id,
                "state_dict": new_weights,
                "delta": delta,
                "num_samples": num_samples,
                "labels": list(cb["label2id"].keys()),
                "discovered": discovered, 
            })

            per_client_metrics.append({
                "id": client_id,
                "acc": float(client_acc),
                "num_samples": int(num_samples)
            })

        log_and_print(f"[SYBIL DEBUG] round={rnd} attackers={list(attacker_ids)}", log_file=run_file)
        
        # Filter just the attackers' current updates
        malicious_updates = {
            cid: upd
            for cid, upd in current_sybil_updates_np
            if cid in attacker_ids
        }

        # Metadata for prover/engine
        client_metadatas = {
            cm["id"]: {"num_samples": cm["num_samples"]}
            for cm in per_client_metrics
        }

        if malicious_updates:
            sybil_updates = {}
            meta = {}

            # Apply sybil logic (This makes them look identical/coordinated)
            for cid, grad in malicious_updates.items():
                out, m = engines.sybil_engine.apply(
                    grad,
                    client_metadata=client_metadatas.get(cid, {})
                )
                sybil_updates[cid] = out
                meta[cid] = m

            for cu in client_updates:
                cid = cu["client_id"]
                if cid in sybil_updates:
                    cu["delta"] = numpy_delta_to_torch(
                        sybil_updates[cid],
                        device=_device_from_state_dict(global_weights),
                        ref_state_dict=global_weights
                    )
                    # Sybils might fake their sample counts too
                    cu["num_samples"] = meta.get(cid, {}).get("num_samples", cu["num_samples"])

            # Audit the attack (Prover)
            if engines.enable_prover and sybil_updates:
                prover = engines.make_sybil_prover(round_id=rnd)
                if prover is not None:
                    prover.observe(
                        benign_update=None,
                        malicious_updates=sybil_updates,
                        client_metadatas=client_metadatas,
                        fake_data_size=meta.get("num_samples")
                    )
                    prover.run()

            # Update shared vector for coordinated attacks
            if run_cfg.sybil_mode in ("leader", "coordinated"):
                engines.sybil_engine.update_shared_vector(
                    [g for _, g in current_sybil_updates_np]
                )

        previous_sybil_updates.append(current_sybil_updates_np)
        if len(previous_sybil_updates) > 5:
            previous_sybil_updates = previous_sybil_updates[-5:]

        public_out = server.run_round(rnd, client_updates)
        
        # Evaluate Global Model
        global_acc = server.evaluate_global(global_test_ds, batch_size=cfg.batch_size)
        log_and_print(f"[GLOBAL] Acc after round {rnd}: {global_acc:.4f}", log_file=run_file)

        # Log Metrics
        rejected_list = []
        if "decisions" in public_out:
            rejected_list = [k for k, v in public_out["decisions"].items() if v == "REJECT"]
        rejected_str = ";".join(rejected_list)

        entry = {
            "round": rnd,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "global_acc": float(global_acc),
            "clients": per_client_metrics,
            "rejected": rejected_list
        }
        
        # JSON Log
        with open(json_path, "r+", encoding="utf-8") as f:
            logs = json.load(f)
            logs.append(entry)
            f.seek(0)
            json.dump(logs, f, indent=2)
            f.truncate()

        # CSV Log
        row = [rnd, entry["timestamp"], float(global_acc)]
        for cm in per_client_metrics:
            row += [cm["acc"], cm["num_samples"]]
        row.append(rejected_str)
        
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    log_and_print("Training completed.", log_file=run_file)

if __name__ == "__main__":
    main()
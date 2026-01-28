import json
import csv
import torch
import numpy as np
import random
from pathlib import Path
from torch.utils.data import DataLoader
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
    log_and_print(f"[ConfigRun] {exp.summary()} attackers={set(run_cfg.attacker_ids)}", log_file=run_file)

    # Build attack engines
    engines = AttackEngines(run_cfg, base_proof_dir=proof_dir)

    # Build dataset & server
    train_base, val_base, test_base, vocab_base, label2id_base = DatasetBuilder.build_dataset(
        path=BASE_DIR / "data" / "animal" / "base" / "base_model.csv",
        max_len=cfg.max_seq_len,
        text_col="Information",
        label_col="Group"
    )

    anchor_loader = DataLoader(train_base, batch_size=cfg.batch_size, shuffle=False)

    server = Server(
        model_cls=ToyBERTClassifier,
        config=cfg,
        device="cuda" if torch.cuda.is_available() else "cpu",
        text_col="Information",
        label_col="Group",
        anchor_loader=anchor_loader,
        checkpoint_dir="checkpoints/base_model"
    )

    # Build clients
    client_paths = [
        BASE_DIR / "data" / "animal" / f"n{i}" / f"client_{i}_data.csv"
        for i in range(1, 13)
    ]

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
        
        # ---- create client object ----
        client_obj = Client(
            client_id=f"client_{i+1}",
            model_fn=make_model_fn(),
            dataset=train_ds,
            device="cuda" if torch.cuda.is_available() else "cpu"
            # Removed KG arguments
        )

        # ---- attach MC-GRAD engine if needed ----
        if run_cfg.mc_grad_train and f"client_{i+1}" in attacker_ids:
            client_obj.mc_grad_engine = engines.mc_grad_engine
        else:
            client_obj.mc_grad_engine = None

        # ---- store client ----
        clients.append({
            "id": f"client_{i+1}",
            "label2id": label2id,
            "client": client_obj,
            "val": val_ds,
            "test": test_ds
        })

    # Global test set
    _, _, global_test_ds, _, _ = DatasetBuilder.build_dataset(
        path=BASE_DIR / "data" / "animal" / "base" / "base_model.csv",
        max_len=cfg.max_seq_len,
        text_col="Information",
        label_col="Group"
    )

    json_path = log_dir / "accuracy_log.json"
    csv_path = log_dir / "accuracy_log.csv"
    if not json_path.exists():
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)

    def write_csv_header(num_clients):
        cols = ["round", "timestamp", "global_acc"]
        for i in range(1, num_clients + 1):
            cols += [f"client_{i}_acc", f"client_{i}_samples"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(cols)

    write_csv_header(len(clients))

    # Federated Learning Loop preparation
    global_weights = server.global_model.state_dict()
    num_rounds = run_cfg.num_rounds
    local_epochs = run_cfg.local_epochs

    previous_updates_for_mc_grad: List[Dict[str, np.ndarray]] = []
    previous_sybil_updates: List[List[Dict[str, np.ndarray]]] = []

    # Main rounds
    for rnd in range(1, num_rounds + 1):
        log_and_print(f"\n[Main] Round {rnd}/{num_rounds} - exp={exp.experiment_case}", log_file=run_file)
        client_updates = []
        per_client_metrics = []
        current_sybil_updates_np = []

        for cb in clients:
            client_obj = cb["client"]
            client_id = cb["id"]

            original_dataset = client_obj.dataset
            fake_num_samples = None

            # --- APPLY DATA ATTACKS ---
            if exp.data_mode:
                try:
                    # Clean call to engine
                    attacked_ds, fake_num_samples = engines.apply_data_attacks(
                        client_dataset=client_obj.dataset,
                        client_id=client_id,
                        round_id=rnd,
                        log_file=run_file,
                        label2id=cb["label2id"]
                    )
                    client_obj.dataset = attacked_ds
                except Exception as e:
                    log_and_print(f"[WARN] DATA-ATTACKS {client_id}: {e}", log_file=run_file)

            # Local training
            new_weights, num_samples, discovered = client_obj.local_train(
                global_weights=global_weights,
                epochs=local_epochs,
                batch_size=cfg.batch_size,
                lr=cfg.lr
            )

            # Restore original dataset
            client_obj.dataset = original_dataset

            # Override sample count if data attack changed it
            if fake_num_samples is not None:
                num_samples = fake_num_samples

            # Local evaluation
            client_acc = client_obj.evaluate(weights=new_weights, batch_size=cfg.batch_size)
            log_and_print(f"[LocalEval] {client_id} acc={client_acc:.4f}", log_file=run_file)

            # Compute delta
            device = _device_from_state_dict(global_weights)
            new_weights = {k: v.to(device) for k, v in new_weights.items()}
            delta = {}
            for k in global_weights.keys():
                try:
                    delta[k] = safe_param_subtract(new_weights[k], global_weights[k])
                except:
                    delta[k] = torch.zeros_like(global_weights[k])

            # Convert to numpy for gradient-level attacks
            delta_np = torch_delta_to_numpy(delta)

            # Store clean copy for MC history
            clean_copy = {k: v.copy() for k, v in delta_np.items()}

            # Decide which gradient Sybil should amplify
            sybil_source = "NONE"
            if not exp.is_sybil_only:
                sybil_source = run_cfg.sybil_use_grad

            # Gather Sybil vector components
            if client_id in attacker_ids:
                if sybil_source == "MC" and exp.is_mc and exp.grad_mode:
                    sybil_grad = delta_np.copy()        # after MC-GRAD (simulated)
                elif sybil_source == "FR" and exp.is_fr and exp.grad_mode:
                    sybil_grad = delta_np.copy()        # after FR-GRAD (simulated)
                else:
                    sybil_grad = clean_copy             # clean or SYBIL_ONLY

                if isinstance(sybil_grad, dict):
                    current_sybil_updates_np.append((client_id, sybil_grad))

            # --- 2. APPLY GRADIENT ATTACKS ---
            if exp.grad_mode:
                delta_np, num_samples = engines.apply_grad_attacks(
                    delta_np=delta_np,
                    client_id=client_id,
                    num_samples=num_samples,
                    prev_updates=previous_updates_for_mc_grad,
                    round_id=rnd,
                    log_file=run_file
                )

            # Update clean history (MC uses clean history)
            previous_updates_for_mc_grad.append(clean_copy)
            if len(previous_updates_for_mc_grad) > engines.mc_grad_engine.history_window:
                previous_updates_for_mc_grad = previous_updates_for_mc_grad[-engines.mc_grad_engine.history_window:]

            # Convert back to torch
            try:
                delta = numpy_delta_to_torch(delta_np, device=device, ref_state_dict=global_weights)
            except Exception:
                delta = {k: torch.zeros_like(global_weights[k]) for k in global_weights.keys()}

            # Reconstruct final state dict
            final_weights = {}
            for k in global_weights.keys():
                final_weights[k] = global_weights[k] + delta[k]

            client_updates.append({
                "client_id": client_id,
                "state_dict": final_weights,
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

        # --- APPLY SYBIL ATTACK (Global Stage) ---
        if exp.is_sybil_only or exp.experiment_case in ("MC_BOTH", "FR_BOTH"):
            log_and_print(f"[SYBIL DEBUG] round={rnd} attackers={list(attacker_ids)}", log_file=run_file)
            
            # Filter just the attackers' current updates
            malicious_updates = {
                cid: upd
                for cid, upd in current_sybil_updates_np
                if cid in attacker_ids
            }

            client_metadatas = {
                cm["id"]: {"num_samples": cm["num_samples"]}
                for cm in per_client_metrics
            }

            if malicious_updates:
                sybil_updates = {}
                meta = {}

                # Apply sybil logic (amplification/collusion)
                for cid, grad in malicious_updates.items():
                    out, m = engines.sybil_engine.apply(
                        grad,
                        client_metadata=client_metadatas.get(cid, {})
                    )
                    sybil_updates[cid] = out
                    meta[cid] = m

                # Replace updates in the main list
                for cu in client_updates:
                    cid = cu["client_id"]
                    if cid in sybil_updates:
                        cu["delta"] = numpy_delta_to_torch(
                            sybil_updates[cid],
                            device=_device_from_state_dict(global_weights),
                            ref_state_dict=global_weights
                        )
                        cu["num_samples"] = meta.get(cid, {}).get("num_samples", cu["num_samples"])

                # -------- SYBIL PROVER --------
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

        # Update sybil shared vector for next round
        if run_cfg.sybil_mode in ("leader", "coordinated"):
            engines.sybil_engine.update_shared_vector(
                [g for _, g in current_sybil_updates_np]
            )

        previous_sybil_updates.append(current_sybil_updates_np)
        if len(previous_sybil_updates) > 5:
            previous_sybil_updates = previous_sybil_updates[-5:]

        # --- SERVER AGGREGATION (With SelfCheck) ---
        public_out = server.run_round(rnd, client_updates)
        global_acc = server.evaluate_global(global_test_ds, batch_size=cfg.batch_size)
        log_and_print(f"[GLOBAL] Acc after round {rnd}: {global_acc:.4f}", log_file=run_file)

        # Logging
        entry = {
            "round": rnd,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "global_acc": float(global_acc),
            "clients": per_client_metrics,
            "trust_summary": public_out.get("trust_summary", {})
        }
        with open(json_path, "r+", encoding="utf-8") as f:
            logs = json.load(f)
            logs.append(entry)
            f.seek(0)
            json.dump(logs, f, indent=2)
            f.truncate()

        row = [rnd, entry["timestamp"], float(global_acc)]
        for cm in per_client_metrics:
            row += [cm["acc"], cm["num_samples"]]
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    log_and_print("Training completed.", log_file=run_file)

if __name__ == "__main__":
    main()
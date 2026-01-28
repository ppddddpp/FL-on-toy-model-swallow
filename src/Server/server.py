import os
import torch
from pathlib import Path
import random
import time
import numpy as np
import json

from EnviromentSetup.trainer.train_base import BaseTrainer
from DataHandler.dataset_builder import DatasetBuilder
from Helpers.configLoader import Config
from Helpers.Helpers import log_and_print

from Framework.SelfCheck import SelfCheckManager

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CKPT_BASELINE = BASE_DIR / "checkpoints" / "base_model"
CKPT_BASELINE.mkdir(parents=True, exist_ok=True)

class Server:
    def __init__(self, model_cls, 
                    config=None, 
                    checkpoint_dir="checkpoints/base_model", device="cpu", 
                    dataset_path=None, ttl_path=None,
                    text_col=None, label_col=None, 
                    log_dir= BASE_DIR / "logs" / "run.txt"
                ):
        
        self.model_cls = model_cls
        self.config = config if config else Config.load(BASE_DIR / "config" / "config.yaml")
        self.device = torch.device(device if isinstance(device, str) else device)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else CKPT_BASELINE
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir

        # Reproducibility
        torch.manual_seed(self.config.seed if hasattr(self.config, 'seed') else 2709)
        np.random.seed(self.config.seed if hasattr(self.config, 'seed') else 2709)
        random.seed(self.config.seed if hasattr(self.config, 'seed') else 2709)

        # Dataset setup for base model
        self.dataset_path = dataset_path or (BASE_DIR / "data" / "animal" / "base" / "base_model.csv")
        self.text_col = text_col or "text"
        self.label_col = label_col or "label"
        self.ttl_path = ttl_path 

        # Load / Train Base Model
        self.global_model, self.base_vocab, self.base_label2id = self._load_or_train_base()
        self.param_key_order = list(self.global_model.state_dict().keys())

        # Ledger & Reputation
        self.reputation_store = {}   
        self.ledger_store = []       

        self.selfcheck = SelfCheckManager(
            global_model=self.global_model,
            log_dir=self.log_dir
        )

        # Dynamic Label Expansion Flags
        self.allow_dynamic_label_expansion = getattr(self.config, 'allow_dynamic_label_expansion', False)
        self.share_label_space = getattr(self.config, 'share_label_space', False)

        log_and_print(f"[Server] Initialized (Sybil Mode). Label expansion: {self.allow_dynamic_label_expansion}", log_file=self.log_dir)

    def _load_or_train_base(self):
        latest_ckpt = self._get_latest_checkpoint()

        train_ds, val_ds, test_ds, vocab, label2id = DatasetBuilder.build_dataset(
            path=self.dataset_path,
            max_len=self.config.max_seq_len,
            text_col=self.text_col,
            label_col=self.label_col
        )

        vocab_size = len(vocab)
        num_classes = len(label2id)
        model = self.model_cls(
            vocab_size=vocab_size,
            num_classes=num_classes,
            d_model=self.config.model_dim,
            nhead=self.config.num_heads,
            num_layers=self.config.num_layers,
            dim_ff=self.config.ffn_dim,
            max_len=self.config.max_seq_len,
            dropout=self.config.dropout
        ).to(self.device)

        if latest_ckpt is None:
            log_and_print("[Server] No base model found, training new base model...", log_file=self.log_dir)
            trainer = BaseTrainer(
                model=model,
                train_dataset=train_ds,
                val_dataset=val_ds,
                test_dataset=test_ds,
                batch_size=self.config.batch_size,
                lr=self.config.lr,
                cfg=self.config,
                use_wandb=False,
                device=str(self.device)
            )
            trainer.train(epochs=self.config.epochs)

            best_pt_path = self.checkpoint_dir / "best.pt"
            
            if best_pt_path.exists():
                latest_ckpt = best_pt_path
            else:
                ckpt_path = self.checkpoint_dir / "epoch_final.pt"
                torch.save(trainer.model.state_dict(), ckpt_path)
                latest_ckpt = ckpt_path

        log_and_print(f"[Server] Loading base model from {latest_ckpt}", log_file=self.log_dir)
        model.load_state_dict(torch.load(latest_ckpt, map_location=self.device))
        model.to(self.device)
        return model, vocab, label2id

    def _get_latest_checkpoint(self):
        ckpts = list(self.checkpoint_dir.glob("*.pt"))
        if not ckpts: return None
        return max(ckpts, key=os.path.getctime)

    def run_round(self, round_id, client_updates):
        """
        Executes one FL round.
        """
        log_and_print(f"\n[Server] --- Round {round_id} starting ---", log_file=self.log_dir)

        # Handle Dynamic Label Expansion (Optional)
        client_label_sets = [set(cu.get("labels", [])) for cu in client_updates if "labels" in cu]
        if self.allow_dynamic_label_expansion and client_label_sets:
            if self.share_label_space:
                self.sync_labels_and_expand_model(client_label_sets)
            else:
                private_label_union = set().union(*client_label_sets)
                current_labels = set(self.base_label2id.keys())
                unseen = private_label_union - current_labels
                if unseen:
                    self.sync_labels_and_expand_model([current_labels | unseen])

        # Format Updates for SelfCheck
        formatted_updates = {}
        
        for cu in client_updates:
            cid = cu["client_id"]
            delta = cu.get("delta", {}) or {}
            
            # Ensure safe tensor format (CPU, float32)
            safe_delta = {}
            for k, v in delta.items():
                if isinstance(v, torch.Tensor):
                    safe_delta[k] = v.detach().cpu().float()
                else:
                    safe_delta[k] = torch.tensor(v, dtype=torch.float32)
            
            formatted_updates[cid] = safe_delta

        decisions, scores, public_out = self.selfcheck.run_round(
            client_updates=formatted_updates,
            round_id=round_id
        )

        # Filter Accepted Updates
        accepted_entries = []
        for cu in client_updates:
            cid = cu["client_id"]
            decision = decisions.get(cid, "REJECT")
            
            if decision == "ACCEPT":
                accepted_entries.append((
                    cu["state_dict"],
                    cu.get("num_samples", 1),
                    1.0
                ))

        # Ledger & Reputation Update
        for cu in client_updates:
            cid = cu["client_id"]
            decision = decisions.get(cid, "REJECT")
            trust_val = 1.0 if decision == "ACCEPT" else 0.0
            
            old_rep = self.reputation_store.get(cid, 0.5)
            new_rep = 0.8 * old_rep + 0.2 * trust_val
            self.reputation_store[cid] = new_rep

            entry = {
                "round": round_id,
                "client_id": cid,
                "decision": decision,
                "trust": trust_val,
                "reputation": new_rep,
                "timestamp": time.time(),
            }
            self.update_ledger(entry)

        # Aggregation
        if accepted_entries:
            new_global = self.aggregate_with_trust(accepted_entries)
            self.safe_load_state_dict(self.global_model, new_global)
            self.save_checkpoint(new_global, round_num=round_id)
            log_and_print(f"[Server] Aggregated {len(accepted_entries)} clients.", log_file=self.log_dir)
        else:
            log_and_print("[Server] CRITICAL: No updates accepted this round. Global model unchanged.", log_file=self.log_dir)

        # Save label map
        save_path = os.path.join(BASE_DIR / "logs", "label2id_dynamic.json")
        with open(save_path, "w") as f:
            json.dump(self.base_label2id, f, indent=2)

        return public_out

    def weighted_fedavg(self, client_updates):
        server_sd = self.global_model.state_dict()
        new_state = {k: torch.zeros_like(v, device=self.device, dtype=v.dtype) for k, v in server_sd.items()}
        total_weight = 0.0

        for _, num, trust in client_updates:
            total_weight += (float(num) * float(trust))

        if total_weight <= 0:
            return server_sd 

        for state, num, trust in client_updates:
            weight = (float(num) * float(trust)) / total_weight
            for k in new_state.keys():
                if k in state:
                    val = state[k].to(self.device)
                    # Basic reshape fallback
                    if val.shape != new_state[k].shape:
                            if val.numel() == new_state[k].numel():
                                val = val.view_as(new_state[k])
                    
                    if val.shape == new_state[k].shape:
                        new_state[k] += val * weight
        return new_state

    def aggregate_with_trust(self, client_updates):
        if not client_updates: return self.global_model.state_dict()
        return self.weighted_fedavg(client_updates)

    def save_checkpoint(self, global_weights, round_num=None):
        if round_num:
            ckpt_path = self.checkpoint_dir / f"round{round_num}.pt"
        else:
            ckpt_path = self.checkpoint_dir / "final.pt"
        torch.save(global_weights, ckpt_path)
        log_and_print(f"[Server] Saved global model -> {ckpt_path}", log_file=self.log_dir)

    def evaluate_global(self, dataset, batch_size=16):
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=batch_size)
        self.global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for ids, mask, y in loader:
                ids, mask, y = ids.to(self.device), mask.to(self.device), y.to(self.device)
                logits = self.global_model(ids, attention_mask=mask)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = float(correct / total) if total > 0 else 0.0
        log_and_print(f"[Server] Global Model Accuracy: {acc:.4f}", log_file=self.log_dir)
        return acc

    def update_ledger(self, entry: dict):
        self.ledger_store.append(entry)
        ledger_path = Path("checkpoints/ledger_log.json")
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            existing = []
            if ledger_path.exists():
                with open(ledger_path, "r") as f:
                    existing = json.load(f)
            existing.append(entry)
            with open(ledger_path, "w") as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            print(f"Ledger error: {e}")

    def safe_load_state_dict(self, model, state_dict):
        model_dict = model.state_dict()
        compatible = {}
        for k, v in state_dict.items():
            if k in model_dict:
                expected = model_dict[k]
                if v.shape == expected.shape:
                    compatible[k] = v.to(expected.device)
                else:
                    pass
        model.load_state_dict(compatible, strict=False)

    def sync_labels_and_expand_model(self, client_label_sets):
        current_labels = set(self.base_label2id.keys())
        new_labels = set().union(*client_label_sets) - current_labels
        if not new_labels: return

        log_and_print(f"[Server] Expanding labels: {new_labels}", log_file=self.log_dir)
        next_id = max(self.base_label2id.values()) + 1
        for label in sorted(new_labels):
            self.base_label2id[label] = next_id
            next_id += 1
        
        num_labels = len(self.base_label2id)
        
        classifier = None
        for name, module in self.global_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                classifier = module 
        
        if classifier:
            old_w = classifier.weight.data
            old_out, old_in = old_w.shape
            
            if old_out < num_labels:
                new_w = torch.zeros((num_labels, old_in), device=self.device)
                new_w[:old_out, :] = old_w
                classifier.weight = torch.nn.Parameter(new_w)
                
                if classifier.bias is not None:
                    old_b = classifier.bias.data
                    new_b = torch.zeros((num_labels,), device=self.device)
                    new_b[:old_out] = old_b
                    classifier.bias = torch.nn.Parameter(new_b)
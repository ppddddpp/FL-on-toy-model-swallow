import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from typing import Any, Dict
from pathlib import Path

from Helpers.Helpers import log_and_print

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Client:
    def __init__(
        self,
        client_id,
        model_fn,
        dataset,
        device="cpu",
        log_dir= BASE_DIR / "logs",
    ):
        self.client_id = client_id
        self.model_fn = model_fn
        self.dataset = dataset
        self.device = device
        
        # Malicious Gradient Engine
        self.mc_grad_engine = None 

        self.log_dir = log_dir / f"client_{self.client_id}.txt"
        self.log_dir.parent.mkdir(parents=True, exist_ok=True)

    def _load_state_safely(self, model, global_weights):
        """
        Safely load global weights into client model.
        Allows classifier head replacement when shape differs (due to label expansion).
        """
        model_dict = model.state_dict()
        new_state = {}

        for k, v in global_weights.items():
            if k in model_dict:
                local_shape = model_dict[k].shape
                global_shape = v.shape

                # classifier shape changes are allowed
                if "classifier" in k and local_shape != global_shape:
                    # Classifier expansion allowed only when global is larger
                    if global_shape[0] > local_shape[0]:
                        log_and_print(f"[SafeLoad] Expanding classifier {k}: {local_shape} -> {global_shape}", log_file=self.log_dir)
                        new_state[k] = v.to(model_dict[k].device)
                    else:
                        log_and_print(f"[SafeLoad] Ignoring smaller classifier {k}, keeping local shape {local_shape}", log_file=self.log_dir)
                        pass
                    continue

                # normal matching case
                if local_shape == global_shape:
                    new_state[k] = v.to(model_dict[k].device)
                else:
                    log_and_print(f"[SafeLoad] Skipped weight '{k}' due to shape mismatch "
                            f"{tuple(local_shape)} vs {tuple(global_shape)}", log_file=self.log_dir)
            else:
                pass

        # update model
        model_dict.update(new_state)
        model.load_state_dict(model_dict)
        return model

    def local_train(
        self,
        global_weights,
        epochs=1,
        batch_size=16,
        lr=1e-3,
        save_path=None,
    ):
        if hasattr(self, "_cached_model"):
            model = self._cached_model.to(self.device)
        else:
            model = self.model_fn().to(self.device)

        model = self._load_state_safely(model, global_weights)
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(epochs):
            for batch_idx, (ids, mask, y) in enumerate(loader):
                ids, mask, y = ids.to(self.device), mask.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                logits = model(ids, attention_mask=mask)
                loss = criterion(logits, y)
                loss.backward()

                # Inject malicious gradient
                if self.mc_grad_engine is not None:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_np = param.grad.detach().cpu().numpy()
                            # Generate malicious gradient
                            attacked = self.mc_grad_engine.generate({name: grad_np})[name]
                            param.grad = torch.from_numpy(attacked).to(param.device)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        # save
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            log_and_print(f"[{self.client_id}] Saved model to {save_path}", log_file=self.log_dir)

        # Update cached global model after local training (for next round)
        self._cached_model = model.to("cpu")
        self._cached_global_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

        discovered = self.discover_local_labels()
        # Return weights on CPU to save GPU memory
        return {k: v.cpu() for k, v in model.state_dict().items()}, len(self.dataset), discovered

    def evaluate(self, weights=None, batch_size=16):
        model = self.model_fn().to(self.device)
        if weights is not None:
            model = self._load_state_safely(model, weights)

        loader = DataLoader(self.dataset, batch_size=batch_size)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for ids, mask, y in loader:
                ids, mask, y = ids.to(self.device), mask.to(self.device), y.to(self.device)
                logits = model(ids, attention_mask=mask)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total if total > 0 else 0.0
        log_and_print(f"[{self.client_id}] Evaluation Accuracy: {acc:.4f}", log_file=self.log_dir)
        return acc

    def discover_local_labels(self) -> Dict[str, Any]:
        """
        Returns a dict with:
            - 'label_ids': set of integer label ids found in the dataset
            - 'label_names': optional set of label name strings if dataset provides mapping
        """
        label_ids = set()
        # try to iterate dataset quickly
        try:
            for _, _, y in DataLoader(self.dataset, batch_size=256):
                label_ids.update([int(v.item()) for v in y])
        except Exception:
            try:
                for sample in self.dataset:
                    y = sample[-1]
                    label_ids.add(int(y.item()) if torch.is_tensor(y) else int(y))
            except Exception:
                pass

        label_names = None
        if hasattr(self.dataset, "id2label"):
            label_names = {self.dataset.id2label[i] for i in label_ids if i in self.dataset.id2label}
        elif hasattr(self, "local_id2label"):
            label_names = {self.local_id2label[i] for i in label_ids if i in self.local_id2label}

        return {"label_ids": label_ids, "label_names": label_names}
    
    def load_global_model(self, global_state: Dict[str, torch.Tensor]):
        """
        Called by server when broadcasting an expanded global model.
        """
        model = self.model_fn().to(self.device)

        try:
            model = self._load_state_safely(model, global_state)
        except Exception as e:
            log_and_print(f"[{self.client_id}] Safe load failed: {e}", log_file=self.log_dir)
            model.load_state_dict(global_state, strict=False)

        # Cache for later local_train
        self._cached_global_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }
        self._cached_model = model
        return True
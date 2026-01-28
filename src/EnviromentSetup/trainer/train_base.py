from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import wandb
from DataHandler import DatasetBuilder
from Helpers import Config
from EnviromentSetup.model.model import ToyBERTClassifier

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

class BaseTrainer:
    def __init__(
        self,
        model: nn.Module = None,
        train_dataset=None,
        val_dataset=None,
        test_dataset=None,
        batch_size: int = 32,
        lr: float = 1e-4,
        device: str = None,
        save_dir: str = "checkpoints/base_model",
        cfg=None,
        use_wandb: bool = False
    ):
        cfg = Config.load(os.path.join(BASE_DIR, "config", "config.yaml")) if cfg is None else cfg
        self.cfg = cfg

        if train_dataset is None:
            print("[BaseTrainer] Building datasets from config...")

            dataset_path = BASE_DIR / "data" / "animal" / "base" / "base_model.csv"
            train_dataset, val_dataset, test_dataset, vocab, label2id = DatasetBuilder.build_dataset(
                path=dataset_path,
                max_len=cfg.max_seq_len,
                text_col="Information",
                label_col="Group"
            )

            vocab_size = len(vocab)
            num_classes = len(label2id)
        else:
            vocab_size = getattr(train_dataset, "vocab_size", None)
            num_classes = getattr(train_dataset, "num_classes", None)

        if model is None:
            model = ToyBERTClassifier(
                vocab_size=vocab_size,
                num_classes=num_classes,
                d_model=cfg.model_dim,
                nhead=cfg.num_heads,
                num_layers=cfg.num_layers,
                dim_ff=cfg.ffn_dim,
                max_len=cfg.max_seq_len,
                dropout=cfg.dropout,
            )

        self.model = model
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

        # --- DataLoaders ---
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size) if test_dataset else None

        # --- Loss and optimizer ---
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # --- Save dir ---
        self.save_dir = Path(save_dir) if save_dir else BASE_DIR / "checkpoints" / "base_model"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # --- W&B logging ---
        self.use_wandb = use_wandb and hasattr(self.cfg, "project_name")

        if self.use_wandb:
            wandb.init(
                project=self.cfg.project_name,
                config={
                    "epochs": self.cfg.epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "model_dim": self.cfg.model_dim,
                    "ffn_dim": self.cfg.ffn_dim,
                    "num_heads": self.cfg.num_heads,
                    "max_seq_len": self.cfg.max_seq_len,
                }
            )
            wandb.watch(self.model, log="all")


    def train(self, epochs: int = 5, save_every: int = 1):
        """
        Train the model for a number of epochs, with optional validation.
        Standard Cross Entropy training (No KG alignment).
        """
        patient = epochs // 5
        best_loss = float("inf")
        
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0

            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}/{epochs}"):
                if len(batch) == 2:
                    input_ids, labels = [x.to(self.device) for x in batch]
                    attention_mask = None
                elif len(batch) == 3:
                    input_ids, attention_mask, labels = [x.to(self.device) for x in batch]
                else:
                    raise ValueError("Unexpected batch format")

                logits = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    return_hidden=False 
                )
                
                loss = self.criterion(logits, labels)

                # --- optimize ---
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f"[BaseTrainer] Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

            if self.use_wandb:
                wandb.log({"train_loss": avg_loss, "epoch": epoch})

            # Validation
            if self.val_loader:
                val_acc = self.evaluate(split="val")
                print(f"[BaseTrainer] Validation Accuracy: {val_acc:.4f}")
                if self.use_wandb:
                    wandb.log({"val_acc": val_acc, "epoch": epoch})

                # Early stopping
                if val_acc > best_loss:
                    best_loss = val_acc
                    patient = epochs // 5
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    patient -= 1
                    if patient == 0:
                        print(f"[BaseTrainer] Early stopping at epoch {epoch}")
                        break

            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch)

    def evaluate(self, split: str = "val"):
        if split == "val" and not self.val_loader:
            raise ValueError("No validation dataset provided")
        if split == "test" and not self.test_loader:
            raise ValueError("No test dataset provided")

        loader = self.val_loader if split == "val" else self.test_loader
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for batch in loader:
                if len(batch) == 2:
                    input_ids, labels = [x.to(self.device) for x in batch]
                    logits = self.model(input_ids)
                else:
                    input_ids, attention_mask, labels = [x.to(self.device) for x in batch]
                    logits = self.model(input_ids, attention_mask)

                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        if self.use_wandb:
            wandb.log({f"{split}_acc": acc})

        return acc

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        ckpt_path = self.save_dir / f"epoch{epoch}.pt" if not is_best else self.save_dir / "best.pt"
        torch.save(self.model.state_dict(), ckpt_path)
        print(f"[BaseTrainer] Saved checkpoint -> {ckpt_path}")

    def load_checkpoint(self, ckpt_path: str):
        ckpt_path = Path(ckpt_path)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.to(self.device)
        print(f"[BaseTrainer] Loaded checkpoint <- {ckpt_path}")
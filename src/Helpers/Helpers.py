import torch
from pathlib import Path
import json
import warnings
import numpy as np
from typing import Dict
import pandas as pd

from DataHandler.dataloader import ToyTextDataset

def log_and_print(*msgs, log_file=None):
    """
    Prints and logs the given messages to the specified log file.

    Args:
        *msgs: The messages to print and log.
        log_file: The path to the log file. Skip log if log file path is none

    Raises:
        ValueError: If log_file is None.
    """
    text = " ".join(str(m) for m in msgs)
    print(text)
    
    if log_file is None:
        warnings.warn("Log file is None skip saving log to file")
        return
    
    log_file = Path(log_file)
    if not log_file.parent.exists():
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def log_round_summary(summary, log_dir="logs"):
    """
    Logs the given round summary as a JSON object to a file in the given log directory.

    Args:
        summary (dict): The round summary to log.
        log_dir (str, optional): The directory to log to. Defaults to "logs".
    """
    if not Path(log_dir).exists():
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    log_path = Path(log_dir) / "round_summary.jsonl"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")

def _device_from_state_dict(sd):
    for v in sd.values():
        if isinstance(v, torch.Tensor):
            return v.device
    return torch.device("cpu")

def torch_delta_to_numpy(delta_torch: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    out = {}
    for k, v in delta_torch.items():
        try:
            out[k] = v.detach().cpu().numpy().copy()
        except Exception:
            out[k] = np.array(v).copy()
    return out

def numpy_delta_to_torch(delta_numpy: Dict[str, np.ndarray], device, ref_state_dict) -> Dict[str, torch.Tensor]:
    out = {}
    for k, arr in delta_numpy.items():
        ref = ref_state_dict[k]
        t = torch.from_numpy(np.array(arr)).to(device=device, dtype=ref.dtype)
        if t.shape != ref.shape:
            try:
                t = t.reshape(ref.shape)
            except Exception:
                t = torch.zeros_like(ref)
        out[k] = t
    return out

def toy_dataset_to_df(ds):
    """Convert ToyTextDataset --> DataFrame(text, label)."""
    return pd.DataFrame({
        "text": ds.texts,
        "label": ds.labels
    })

def df_to_toy_dataset(df, original_ds):
    ds = ToyTextDataset(
        texts=df["Information"].tolist(),
        labels=df["Group"].tolist(),
        vocab=original_ds.vocab,
        max_len=original_ds.max_len,
        num_classes=original_ds.num_classes
    )
    ds.text_col = getattr(original_ds, "text_col", "Information")
    ds.label_col = getattr(original_ds, "label_col", "Group")
    ds.df = df

    return ds

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_json(obj, path):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_json_safe(obj), f, indent=2)

def save_csv(df, path):
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)

def to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj

def flatten_state_dict_to_tensor(state_dict):
    """
    Flattens a PyTorch state_dict (or any dict of tensors) into a single 1D tensor.
    Useful for computing global norms or cosine similarity between updates.
    """
    # Sort keys to ensure consistent ordering every time
    keys = sorted(state_dict.keys())
    
    # Flatten each tensor and collect them
    tensors = []
    for key in keys:
        tensor = state_dict[key]
        # Ensure it's a float tensor (sometimes buffers are int/long)
        if not tensor.is_floating_point():
            tensor = tensor.float()
        tensors.append(tensor.view(-1)) # Flatten to 1D
        
    # Concatenate all into one giant vector
    if not tensors:
        return torch.tensor([])
        
    return torch.cat(tensors)
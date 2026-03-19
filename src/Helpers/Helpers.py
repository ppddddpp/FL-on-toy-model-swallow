import torch
from pathlib import Path
import json
import warnings
import numpy as np
from typing import Dict


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


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_json(obj, path):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_json_safe(obj), f, indent=2)

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

def flatten(grad):
    return np.concatenate([v.flatten() for v in grad.values()])

def average_updates(updates):
    keys = list(next(iter(updates.values())).keys())
    avg = {}
    for k in keys:
        avg[k] = np.mean([u[k] for u in updates.values()], axis=0)
    return avg
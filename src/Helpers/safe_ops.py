import torch
import numpy as np

def safe_param_subtract(a, b):
    """
    Safely compute (a - b) for model parameters or gradients.

    Supports:
        - torch.Tensor
        - numpy.ndarray
        - mixed torch / numpy
        - None or missing values

    Failure policy:
        - return zeros_like(b)
    """

    if a is None or b is None:
        return torch.zeros_like(b) if torch.is_tensor(b) else np.zeros_like(b)

    try:
        # --- Torch tensors ---
        if torch.is_tensor(a) and torch.is_tensor(b):
            if a.shape != b.shape:
                return torch.zeros_like(b)
            if torch.isnan(a).any() or torch.isnan(b).any():
                return torch.zeros_like(b)
            return a - b

        # --- NumPy arrays ---
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            if a.shape != b.shape:
                return np.zeros_like(b)
            if np.isnan(a).any() or np.isnan(b).any():
                return np.zeros_like(b)
            return a - b

        # --- Mixed torch / numpy ---
        if torch.is_tensor(a) and isinstance(b, np.ndarray):
            b_t = torch.from_numpy(b).to(a.device)
            return safe_param_subtract(a, b_t)

        if isinstance(a, np.ndarray) and torch.is_tensor(b):
            a_t = torch.from_numpy(a).to(b.device)
            return safe_param_subtract(a_t, b)

    except Exception:
        pass

    # --- Fail closed ---
    return torch.zeros_like(b) if torch.is_tensor(b) else np.zeros_like(b)

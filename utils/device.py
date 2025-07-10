import torch

def get_default_device():
    """Return the best available torch device (MPS, CUDA or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

import torch


def get_device (preferred_devices = None):
    if preferred_devices is None:
        preferred_devices = ["cuda", "mps", "cpu"]

    for preferred in preferred_devices:
        if preferred == "cuda" and torch.cuda.is_available ():
            return torch.device ("cuda")

        if preferred == "mps" and torch.backends.mps.is_available ():
            return torch.device ("mps")

        if preferred == "cpu":
            return torch.device ("cpu")

    return torch.device ("cpu")
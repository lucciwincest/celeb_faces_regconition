import torch

@torch.no_grad ()
def accuracy (output, target):
    pred = output.argmax (1)
    return (pred == target).sum ().item ()
from pathlib import Path
import torch

class CheckpointManager:
    def __init__ (self, checkpoint_dir):
        self.checkpoint_dir = Path (checkpoint_dir)
        self.checkpoint_dir.mkdir (exist_ok = True)

    def save (self, name, model, optimizer = None, scheduler = None, epoch = 0, best_acc = 0):
        path = self.checkpoint_dir / name
        torch.save ({
            "epoch": epoch,
            "model": model.state_dict (),
            "optimizer": None if optimizer is None else optimizer.state_dict (),
            "scheduler": None if scheduler is None else scheduler.state_dict (),
            "best_acc": best_acc
        })
    
    def load (self, name, model, optimizer = None, scheduler = None, device = "cpu"):
        path = self.checkpoint_dir / name
        checkpoint = torch.load (path, map_location = device)
        model.load_state_dict (checkpoint ["model"])
        if optimizer is not None and optimizer is not None:
            optimizer.load_state_dict (checkpoint ["optimizer"])
        if scheduler is not None and checkpoint ["scheduler"] is not None:
            scheduler.load_state_dict (checkpoint ["scheduler"])
        print ("Checkpoint loaded")
        return {
            "epoch": checkpoint ["epoch"],
            "best_acc": checkpoint ["best_acc"]
        }
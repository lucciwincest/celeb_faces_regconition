from pathlib import Path
import torch


class CheckpointManager:

    def __init__ (self, experiment_dir):
        self.checkpoint_dir = Path (experiment_dir) / "checkpoints"
        self.checkpoint_dir.mkdir (parents = True, exist_ok = True)

    def save (self, filename, model, optimizer = None, scheduler = None, epoch = 0, best_val_acc = None):
        torch.save ({
            "epoch": epoch,
            "model": model.state_dict (),
            "optimizer": None if optimizer is None else optimizer.state_dict (),
            "scheduler": None if scheduler is None else scheduler.state_dict (),
            "best_val_acc": best_val_acc
        }, self.checkpoint_dir / filename)

    def load (self, filename, model, optimizer = None, scheduler = None, device = "cpu"):
        path = self.checkpoint_dir / filename

        if not path.exists ():
            return None

        checkpoint = torch.load (path, map_location = device)

        model.load_state_dict (checkpoint ["model"])

        if optimizer is not None and checkpoint ["optimizer"] is not None:
            optimizer.load_state_dict (checkpoint ["optimizer"])

        if scheduler is not None and checkpoint ["scheduler"] is not None:
            scheduler.load_state_dict (checkpoint ["scheduler"])

        print ("Checkpoint loaded")

        return checkpoint

    def exists (self, filename):
        return (self.checkpoint_dir / filename).exists ()
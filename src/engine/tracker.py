from pathlib import Path
import json


class Tracker:

    def __init__ (self, experiment_dir):
        self.experiment_dir = Path (experiment_dir)
        self.experiment_dir.mkdir (parents = True, exist_ok = True)
        self.history = []

    def log (self, epoch, train_loss, val_acc, lr):
        self.history.append ({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "lr": lr
        })

    def save (self):
        with open (self.experiment_dir / "history.json", "w", encoding = "utf-8") as f:
            json.dump (self.history, f, indent = 2)
from pathlib import Path
import torch
from tqdm import tqdm

class Trainer:

    def __init__ (self, model, train_loader, val_loader, criterion, optimizer, device, scheduler = None, checkpoint_dir = "checkpoints"):
        self.model = model.to (device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path (checkpoint_dir)
        self.checkpoint_dir.mkdir (exist_ok = True)

        self.best_acc = 0
        self.start_epoch = 0

    def train_one_epoch (self):
        self.model.train ()
        total_loss = 0
        for X, y in tqdm (self.train_loader, desc = "Training"):
            X = X.to (self.device)
            y = y.to (self.device)

            self.optimizer.zero_grad ()
            Output = self.model (X)
            loss = self.criterion (Output, y)
            loss.backward ()
            self.optimizer.step ()
            total_loss += loss.item ()

        return total_loss / len (self.train_loader)
    
    @torch.no_grad ()
    def validate (self):
        self.model.eval ()

        correct = 0
        total = 0

        for X, y in tqdm (self.val_loader, desc = "Validation"):
            X = X.to (self.device)
            y = y.to (self.device)

            Output = self.model (X)
            pred = Output.argmax (dim = 1)

            correct += (pred == y).sum ().item ()
            total += y.shape [0]

        acc = correct / total

        return acc
    
    def fit (self, epochs):
        for epoch in range (self.start_epoch, epochs):
            train_loss = self.train_one_epoch ()
            val_acc = self.validate ()

            print (f"Epoch {epoch} / {epochs}")
            print (f"loss: {train_loss:.4f}")
            print (f"val acc: {val_acc:.4f}")

            if self.scheduler is not None:
                self.scheduler.step ()

            self.save_checkpoint ("latest.pth", epoch)

    def save_checkpoint (self, name, epoch):
        path = self.checkpoint_dir / name
        torch.save ({
            "epoch": epoch,
            "model": self.model.state_dict (),
            "optimizer": self.optimizer.state_dict (),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict (),
            "best_acc": self.best_acc
        }, path)

    def load_checkpoint (self, path):
        checkpoint = torch.load (path, map_location = self.device)
        self.model.load_state_dict (checkpoint ["model"])
        self.optimizer.load_state_dict (checkpoint ["optimizer"])
        if self.scheduler is not None and checkpoint ["scheduler"] is not None:
            self.scheduler.load_state_dict (checkpoint ["scheduler"])
        self.start_epoch = checkpoint ["epoch"] + 1
        self.best_acc = checkpoint ["best_acc"]
        print ("Checkpoint loaded")

    @torch.no_grad ()
    def test (self, test_loader):
        self.model.eval ()
        
        correct = 0
        total = 0

        for X, y in test_loader:
            X = X.to (self.device)
            y = y.to (self.device)

            Output = self.model (X)
            pred = Output.argmax (dim = 1)

            correct += (pred == y).sum ().item ()
            total += y.shape [0]

        acc = correct / total

        print (f"Test accuracy: {acc:.4f}")

        return acc
    
    @torch.no_grad ()
    def predict (self, X):
        self.model.eval ()
        X = X.to (self.device)
        Output = self.model (X)
        pred = Output.argmax (dim = 1)
        return pred


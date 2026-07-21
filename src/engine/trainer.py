import torch


class Trainer:

    def __init__ (self, model, train_loader, val_loader, criterion, optimizer, device, scheduler = None):
        self.model = model.to (device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_one_epoch (self):
        self.model.train ()
        total_loss = 0

        for X, y in self.train_loader:
            X = X.to (self.device)
            y = y.to (self.device)
            self.optimizer.zero_grad  ()
            output = self.model (X)
            loss = self.criterion (output, y)
            loss.backward ()
            self.optimizer.step ()
            total_loss += loss.item ()

        return total_loss / len(self.train_loader)

    @torch.no_grad ()
    def validate (self):
        self.model.eval ()
        correct = 0
        total = 0

        for X, y in self.val_loader:
            X = X.to (self.device)
            y = y.to (self.device)
            output = self.model (X)
            pred = output.argmax (dim=1)
            correct += (pred == y).sum ().item ()
            total += y.size (0)

        return correct / total

    def fit (self, start_epoch, epochs):

        for epoch in range (start_epoch, epochs):
            train_loss = self.train_one_epoch ()
            val_acc = self.validate ()
            print (f"Epoch {epoch}/{epochs}")
            print (f"Loss: {train_loss:.4f}")
            print (f"Val Acc: {val_acc:.4f}")

            if self.scheduler is not None:
                self.scheduler.step (val_acc)

        return self.model
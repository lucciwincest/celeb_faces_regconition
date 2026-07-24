import torch


class Trainer:

    def __init__ (self, model, train_loader, val_loader, criterion, optimizer, device, scheduler = None, tracker = None, checkpoint_manager = None, callbacks = None):
        self.model = model.to (device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tracker = tracker
        self.checkpoint_manager = checkpoint_manager
        self.callbacks = callbacks or []
        self.device = device

        self.best_val_acc = 0
        self.current_epoch = 0
        self.val_acc = 0
        self.stop_training = False

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

        return total_loss / len (self.train_loader)

    @torch.no_grad ()
    def validate (self):
        self.model.eval ()
        correct = 0
        total = 0

        for X, y in self.val_loader:
            X = X.to (self.device)
            y = y.to (self.device)
            output = self.model (X)
            pred = output.argmax (dim = 1)
            correct += (pred == y).sum ().item ()
            total += y.size (0)

        return correct / total

    def fit (self, start_epoch, end_epoch):
        self.stop_training = False

        for callback in self.callbacks:
            callback.on_train_begin (self)

        for epoch in range (start_epoch, end_epoch):
            self.current_epoch = epoch
            for callback in self.callbacks:
                callback.on_epoch_begin (self)

            train_loss = self.train_one_epoch ()
            self.val_acc = self.validate ()
            print (f"Epoch {epoch}/{end_epoch}")
            print (f"Loss: {train_loss:.4f}")
            print (f"Validation Accuracy: {self.val_acc:.4f}")

            lrs = [group ["lr"] for group in self.optimizer.param_groups]

            if self.tracker is not None:
                self.tracker.log (epoch, train_loss, self.val_acc, lrs)

            if self.checkpoint_manager is not None:
                self.checkpoint_manager.save ("last.pth", self.model, self.optimizer, self.scheduler, epoch, self.best_val_acc)
                if self.val_acc > self.best_val_acc:
                    self.best_val_acc = self.val_acc
                    self.checkpoint_manager.save ("best.pth", self.model, self.optimizer, self.scheduler, epoch, self.best_val_acc)

            if self.scheduler is not None:
                self.scheduler.step (self.val_acc)

            for callback in self.callbacks:
                callback.on_epoch_end (self)

            if self.stop_training:
                print ("Early stopping")
                break

        if self.tracker is not None:
            self.tracker.save ()

        for callback in self.callbacks:
            callback.on_train_end (self)

    def resume (self):
        if self.checkpoint_manager is None:
            return 0
        if not self.checkpoint_manager.exists ("last.pth"):
            return 0
        checkpoint = self.checkpoint_manager.load ("last.pth", self.model, self.optimizer, self.scheduler, self.device)
        self.best_val_acc = checkpoint ["best_val_acc"]
        return checkpoint ["epoch"] + 1

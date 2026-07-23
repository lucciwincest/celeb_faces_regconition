from .callback import Callback


class EarlyStopping (Callback):

    def __init__ (self, patience = 5):
        self.patience = patience
        self.counter = 0
        self.best = - float ("inf")

    def on_epoch_end (self, trainer):
        if trainer.val_acc > self.best:
            self.best = trainer.val_acc
            self.counter = 0

        else:
            self.counter += 1
            if self.counter >= self.patience:
                trainer.stop_training = True
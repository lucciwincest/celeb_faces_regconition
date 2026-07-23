from .callback import Callback
from src.utils.freeze import freeze, unfreeze


class ProgressiveUnfreezer (Callback):

    def __init__ (self, backbone, schedule):
        self.backbone = backbone
        self.schedule = schedule

    def on_train_begin (self, trainer):
        freeze (self.backbone)

    def on_epoch_begin (self, trainer):

        epoch = trainer.current_epoch

        if epoch not in self.schedule:
            return

        for name in self.schedule [epoch]:

            if name == "all":
                unfreeze (self.backbone)
            else:
                unfreeze (getattr (self.backbone, name))
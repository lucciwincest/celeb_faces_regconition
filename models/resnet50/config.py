from dataclasses import dataclass, field


@dataclass (slots = True)
class Config:

    epochs = 30

    batch_size = 32

    learning_rate = 1e-4

    weight_decay = 1e-4

    freeze_backbone = True

    progressive_unfreezing = True

    unfreeze_schedule = field (
        default_factory = lambda: {
            5: ["layer4"],
            10: ["layer3"],
            15: ["layer2"],
            20: ["all"]
        }
    )

    experiment_dir = "models/resnet50/experiments"


cfg = Config ()
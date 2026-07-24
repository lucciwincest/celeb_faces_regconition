from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    data_root: Path = Path("dataset")
    experiment_dir: Path = Path("models/resnet50/experiments")

    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-4

    seed: int = 42
    patience: int = 5
    image_size: tuple [int, int] = (224, 224)

    prefer_devices: list[str] = field(
        default_factory=lambda: ["cuda", "mps", "cpu"]
    )

    resume: bool = False

    freeze_backbone: bool = True
    progressive_unfreezing: bool = True

    unfreeze_schedule: dict[int, list[str]] = field(
        default_factory=lambda: {
            5: ["layer4"],
            10: ["layer3"],
            15: ["layer2"],
            20: ["all"],
        }
    )


cfg = Config()
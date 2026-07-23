from  pathlib import Path

DATASET_ROOT = Path ("../dataset")

IN_CHANNELS = 3
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 23

EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3

DEVICE = "cuda"

CHECKPOINT_DIR = Path ("../checkpoints")
BEST_MODEL = CHECKPOINT_DIR / "best.pth"
LATEST_MODEL = CHECKPOINT_DIR / "latest.pth"
from pathlib import Path
import sys
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Detect if running on Google Colab
try:
    from google.colab import drive
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Set project root based on environment
# On Colab, project is cloned to /content/ai-image-detector
# On local machine, it's the current directory
if IS_COLAB:
    PROJECT_ROOT = Path('/content/ai-image-detector')
else:
    PROJECT_ROOT = ROOT

DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "valid"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PLOT_DIR = OUTPUT_DIR / "plots"
GRADCAM_DIR = OUTPUT_DIR / "gradcam"

IMG_SIZE = 380
BATCH_SIZE = 18
NUM_WORKERS = 2
NUM_CLASSES = 2
CLASS_NAMES = ["REAL", "FAKE"]

MODEL_NAME = "efficientnet_b4"
PRETRAINED = True
DROPOUT = 0.4

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 12
PATIENCE = 5

SEED = 42

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()

for path in [OUTPUT_DIR, CHECKPOINT_DIR, PLOT_DIR, GRADCAM_DIR]:
    path.mkdir(parents=True, exist_ok=True)

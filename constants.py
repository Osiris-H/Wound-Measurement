import torch
from pathlib import Path

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

MODELS_DIR = Path.cwd() / "models"
CKPT_DIR = Path.cwd() / "ckpt"
DATA_ROOT = Path.cwd().parents[0] / "Data"
EVAL_RESULT_ROOT = Path.cwd().parents[0] / "Results" / "eval"

RESNET34_CKPT_PATH = CKPT_DIR / "resnet34-b627a593.pth"
FNET_CKPT_PATH = CKPT_DIR / "model_099_0.9588.pth.tar"
SAM2_BASE_PLUS_CKPT_PATH = CKPT_DIR / "sam2.1_hiera_base_plus.pt"

SAM2_BASE_PLUS_CONFIG_PATH = MODELS_DIR / "sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"

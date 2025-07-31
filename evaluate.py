import torch
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from utils.plot import overlay_mask_edge
from inference import fnet_inference

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

data_root = Path.cwd().parents[0] / "Data"
output_dir = Path.cwd().parents[0] / "Results" / "eval" / "New"
image_dir = data_root / "New" / "images"

''''''
image_paths = list(image_dir.iterdir())
with tqdm(image_paths, unit="img") as pbar:
    for image_path in pbar:
        if not image_path.is_file():
            continue

        pil_image = Image.open(image_path).convert("RGB")
        pred_mask = fnet_inference(pil_image)
        out_image = overlay_mask_edge(pil_image, pred_mask)
        # out_image.show()
        out_image.save(output_dir / f"{image_path.stem}.png", format="PNG")
        # break


# for image_path in output_dir.iterdir():
#     filename = image_path.stem
#     new_path = output_dir / f"{filename}_fnet.png"
#     os.rename(image_path, new_path)

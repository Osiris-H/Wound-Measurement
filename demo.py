import re

import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from utils.common import setup_logger
from utils.prompts import mask_to_boxes
from inference import fnet_inference, sam2_inference
from utils.evaluate import *

image_dir = Path.cwd().parents[0] / "Data" / "FUSeg" / "test" / "images"
mask_dir = Path.cwd().parents[0] / "Data" / "FUSeg" / "test" / "labels"
log_dir = Path.cwd().parents[0] / "Results" / "eval" / "FUSeg"
demo_dir = Path.cwd().parents[0] / "Data" / "Demo"


def overlay_mask_edge(
        rgb_image: np.ndarray,
        mask: np.ndarray,
        color: tuple[int, int, int] = (0, 255, 0),  # lime‐green for medical overlays
        thickness: int = 1
) -> np.ndarray:
    """
    Draws the external contour of a boolean mask onto an RGB image.

    Args:
      rgb_image: H×W×3 uint8 array in RGB order.
      mask:      H×W boolean or uint8 array (True/1 = foreground).
      color:     (R, G, B) color for the edge (default is light blue).
      thickness: line thickness in pixels.

    Returns:
      A copy of `rgb_image` with the mask edge overlaid.
    """
    # 1) ensure mask is uint8 binary
    bin_mask = (mask.astype(bool)).astype(np.uint8) * 255

    # 2) find external contours (no approximation → full fidelity)
    contours, _ = cv2.findContours(
        bin_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    # 3) convert RGB → BGR for OpenCV drawing
    overlay = rgb_image.copy()
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    # 4) draw all contours in BGR order
    color_bgr = (color[2], color[1], color[0])
    cv2.drawContours(
        overlay_bgr,
        contours,
        contourIdx=-1,  # draw all
        color=color_bgr,
        thickness=thickness,
        lineType=cv2.LINE_AA
    )

    # 5) convert back to RGB
    return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)


def show_mask_edge(
        image_path: Path,
        mask_path: Path,
):
    image_name = image_path.stem
    ext = image_path.suffix
    if ext == '.png':
        pil_image = Image.open(image_path).convert("RGB")
        pil_image = ImageOps.exif_transpose(pil_image)
    elif ext == ".jpg":
        pil_image = Image.open(image_path).convert("RGB")
    else:
        print("Image type not supported.")
        return
    image_np = np.array(pil_image)

    assert mask_path.suffix == ".png", "Mask type not supported."
    pil_mask = Image.open(mask_path).convert("L")
    mask_np = np.array(pil_mask)
    true_mask = mask_np > 127
    boxes = mask_to_boxes(mask_np)

    fnet_mask = fnet_inference(pil_image)
    sam2_mask = sam2_inference(pil_image, boxes=boxes)

    for name, pred in [("FNet", fnet_mask), ("SAM2", sam2_mask)]:
        iou = compute_iou(pred, true_mask)
        dsc = compute_dsc(pred, true_mask)
        sens = compute_sens(pred, true_mask)
        spec = compute_spec(pred, true_mask)
        # print(f"{image_name} with {name}: IoU={iou:.4f}, Dice={dsc:.4f}, "
        #       f"Sens={sens:.4f}, Spec={spec:.4f}")
        logger.info(
            f"{image_name} with {name}: IoU={iou:.4f}, Dice={dsc:.4f}, Sens={sens:.4f}, Spec={spec:.4f}"
        )

    # Plot mask edge on original image
    fnet_out = overlay_mask_edge(image_np, fnet_mask)
    sam2_out = overlay_mask_edge(image_np, sam2_mask)

    # Save overlay results
    fnet_out_path = demo_dir / f"FUSeg-{image_name}_fnet_{ext}"
    fnet_out_path.parent.mkdir(exist_ok=True)
    Image.fromarray(fnet_out).save(fnet_out_path)
    sam2_out_path = demo_dir / f"FUSeg-{image_name}_sam2_{ext}"
    sam2_out_path.parent.mkdir(exist_ok=True)
    Image.fromarray(sam2_out).save(sam2_out_path)

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # ax1.imshow(fnet_out)
    # ax1.set_title("FNet Overlay")
    # ax1.axis("off")
    # ax2.imshow(sam2_out)
    # ax2.set_title("SAM2 Overlay")
    # ax2.axis("off")
    # plt.tight_layout()
    # plt.show()


def collect_demo_images():
    log_path = log_dir / "fnet.log"

    pattern = re.compile(
        r"""
        ^\d{4}-\d{2}-\d{2}\s+          # date YYYY-MM-DD
        \d{2}:\d{2}:\d{2}\s+           # time HH:MM:SS
        INFO:\s+                       # literal INFO:
        (?P<name>[^:]+):\s+            # filename (up to next colon)
        IoU=(?P<iou>[0-9]*\.?[0-9]+)   # IoU value
        """,
        re.VERBOSE
    )

    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue

            name, iou = m.groups()
            iou = float(iou)
            if iou < 0.8:
                print(name, iou)
                image_path = image_dir / name
                mask_path = mask_dir / name

                show_mask_edge(image_path, mask_path)

                # break


if __name__ == '__main__':
    logger = setup_logger(demo_dir / "demo.log")
    collect_demo_images()

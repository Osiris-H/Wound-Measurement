import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from pathlib import Path
from models.fnet import FNet
from models.sam2.build_sam import build_sam2
from constants import *
from models.sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.prompts import combine_masks
from typing import Optional


def load_fnet():
    model = FNet()
    model = nn.DataParallel(model)
    ckpt = torch.load(FNET_CKPT_PATH, map_location='cpu', weights_only=False)
    # print(ckpt["state_dict"].keys())
    model.load_state_dict(ckpt["state_dict"])

    return model


def load_sam2(
        config_path: str | Path,
        ckpt_path: str | Path,
        device=DEVICE
):
    """
    Load the SAM2 model from a config file and checkpoint.

    Args:
        config_path (str | Path): Path to the model configuration file.
            Can be provided as either a string or a pathlib.Path.
        ckpt_path (str | Path): Path to the model checkpoint file.
            Can be provided as either a string or a pathlib.Path.
        device: Device identifier (e.g., "cpu" or "cuda"). Defaults to global DEVICE.

    Returns:
        sam2_model: The initialized SAM2 model, built with the given config and checkpoint.
    """
    # Normalize inputs to consistent string paths
    config_path = Path(config_path).as_posix()
    ckpt_path = Path(ckpt_path).as_posix()

    sam2_model = build_sam2(config_path, ckpt_path, device=device, mode="eval")

    return sam2_model


def fnet_inference(pil_image: Image.Image) -> np.ndarray:
    """
    Perform inference using the FNet model on a PIL image.

    Args:
        pil_image (Image.Image): Input PIL image to process.

    Returns:
        np.ndarray: Binary mask as a boolean numpy array where True indicates
                   foreground pixels and False indicates background pixels.
                   Shape matches the original image dimensions (H, W).
    """
    model = load_fnet()
    model = model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),  # → FloatTensor in [0,1], shape C×H×W
    ])

    # (1, C, 512, 512)
    img_tensor = transform(pil_image).unsqueeze(0)

    W, H = pil_image.size
    with torch.no_grad():
        out_logits, _ = model(img_tensor)
        pred = torch.sigmoid(out_logits)
        pred = F.interpolate(pred, size=(H, W), mode="area")
        pred = pred.squeeze().cpu().numpy()

    return pred > 0.5


def sam2_inference(
        pil_image: Image.Image,
        pos_points: list[list[tuple[int, int]]] = None,
        neg_points: list[list[tuple[int, int]]] = None,
        boxes: list[tuple[int, int, int, int]] = None
) -> Optional[np.ndarray]:
    sam2_model = load_sam2(SAM2_BASE_PLUS_CONFIG_PATH, SAM2_BASE_PLUS_CKPT_PATH)
    predictor = SAM2ImagePredictor(sam2_model)

    image_np = np.array(pil_image)
    predictor.set_image(image_np)

    tasks = []

    if pos_points:
        for pts in pos_points:
            coords = np.array(pts)
            labels = np.ones(len(pts), dtype=int)
            tasks.append((coords, labels, None))

    if boxes:
        if pos_points and len(pos_points) != len(boxes):
            print("ERROR: pos_points length must equal boxes length")
            # logger.error("pos_points length must equal boxes length")
            return None
        if neg_points and len(neg_points) != len(boxes):
            print("ERROR: neg_points length must equal boxes length")
            # logger.error("neg_points length must equal boxes length")
            return None
        for idx, box in enumerate(boxes):
            pos_pt = pos_points[idx] if pos_points else []
            neg_pt = neg_points[idx] if neg_points else []

            coord_chunks = []
            label_chunks = []

            if pos_pt:
                coord_chunks.append(np.array(pos_pt))
                label_chunks.append(np.ones(len(pos_pt), dtype=int))
            if neg_pt:
                coord_chunks.append(np.array(neg_pt))
                label_chunks.append(np.zeros(len(neg_pt), dtype=int))

            if coord_chunks:
                coords = np.concatenate(coord_chunks, axis=0)
                labels = np.concatenate(label_chunks, axis=0)
            else:
                coords = None
                labels = None
                tasks.append((coords, labels, np.array(box)))

    if not tasks:
        H, W = image_np.shape[:2]
        return np.zeros((H, W), dtype=bool)

    mask_list = []
    for point_coords, point_labels, box_coords in tasks:
        mask, _, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_coords,
            multimask_output=False,
            return_logits=False,
        )
        # mask.shape = (1, H, W)
        mask_list.append(mask[0])

    if len(mask_list) == 1:
        return mask_list[0] > 0.5

    masks = np.stack(mask_list, axis=0) > 0.5
    return combine_masks(masks)

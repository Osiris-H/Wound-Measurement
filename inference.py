import re
import cv2
import numpy as np
import torch
import logging
import torch.nn.functional as F
from typing import Optional
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm
from pathlib import Path
from fnet import FNet
from PIL import Image, ImageOps, ImageDraw, ImageFile
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.common import *
from utils.prompts import *
from utils.evaluate import *


if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

data_dir = Path.cwd().parents[0] / "Data"
misc_dir = data_dir / "MISC"
fuseg_dir = data_dir / "FUSeg"
log_dir = Path.cwd().parents[0] / "Results" / "eval"


def load_fnet():
    ckpt_path = "ckpt/model_099_0.9588.pth.tar"
    model = FNet()
    model = nn.DataParallel(model)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    # print(ckpt["state_dict"].keys())
    model.load_state_dict(ckpt["state_dict"])

    return model


def load_sam2(device="cpu"):
    sam2_checkpoint = "ckpt/sam2.1_hiera_base_plus.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

    return sam2_model


def combine_masks(masks: np.ndarray) -> np.ndarray:
    """
    Given a list of binary masks for the same image,
    return their pixel-wise OR (union) as one mask.
    """
    assert masks.ndim == 3
    combined = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        combined |= m.astype(bool)
    return combined


def show_mask_gray(mask, ax):
    """
    mask: 2D array of 0s and 1s (or floats in [0,1])
    ax:   a Matplotlib Axes
    """
    ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")


def show_boxes(boxes, ax):
    for box in boxes:
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(
            plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
        )


def show_points(
        points: list[list[tuple[int, int]]],
        ax: plt.Axes,
        color: str = "red",
        size: int = 50,
        marker: str = "o"
):
    """
    Scatter all (x,y) points in a nested list `points` onto `ax`.
    `points` can be either:
      - a flat list of (x,y) tuples, or
      - a list of lists of (x,y) tuples.
    """
    # Normalize to a flat list of tuples
    if not points:
        return

    # If the first element is itself a tuple, assume flat; else flatten
    if isinstance(points[0], tuple):
        flat_pts = points  # type: ignore
    else:
        flat_pts = [pt for sublist in points for pt in sublist]

    if not flat_pts:
        return

    xs, ys = zip(*flat_pts)
    ax.scatter(
        xs, ys,
        c=color,
        s=size,
        marker=marker,
        zorder=3
    )


def fnet_inference(pil_image):
    ckpt_path = "ckpt/model_099_0.9588.pth.tar"
    model = FNet()
    model = nn.DataParallel(model)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    # print(ckpt["state_dict"].keys())
    model.load_state_dict(ckpt["state_dict"])
    ''''''
    model = model.to(device)
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
        # print(pred.shape, pred.dtype)
        # predict = torch.round(torch.sigmoid(main_out)).byte()
        # pred_seg = predict.data.cpu().numpy() * 255

    return pred > 0.5


def sam2_inference(
        pil_image: Image.Image,
        pos_points: list[list[tuple[int, int]]] = None,
        neg_points: list[list[tuple[int, int]]] = None,
        boxes: list[tuple[int, int, int, int]] = None
) -> Optional[np.ndarray]:
    sam2_checkpoint = "ckpt/sam2.1_hiera_base_plus.pt"
    # sam2_checkpoint = "ckpt/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    # model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
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
            logger.error("pos_points length must equal boxes length")
            return None
        if neg_points and len(neg_points) != len(boxes):
            logger.error("neg_points length must equal boxes length")
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


def sam2_from_fnet(
        pil_image: Image.Image,
        boxes: list[tuple[int, int, int, int]]
) -> np.ndarray:
    image_np = np.array(pil_image)
    H, W = image_np.shape[:2]

    # Early exit if no boxes
    if not boxes:
        return np.zeros((H, W), dtype=bool)


    fnet_model = load_fnet().to(device)
    fnet_model.eval()
    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),  # → FloatTensor in [0,1], shape C×H×W
    ])

    # (1, C, 512, 512)
    img_tensor = transform(pil_image).unsqueeze(0)
    with torch.no_grad():
        out_logits, _ = fnet_model(img_tensor)
        low_res = F.interpolate(
            out_logits,
            size=(256, 256),
            mode='bilinear',
            align_corners=True
        )
        # Required by sam2
        low_res = torch.clamp(low_res, -32.0, 32.0)
        low_res_masks = low_res.cpu().numpy()

    sam2_model = load_sam2()
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image_np)

    box_coords = np.array(boxes)
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box_coords,
        mask_input=low_res_masks,
        multimask_output=False,
        return_logits=False,
    )

    if masks.ndim == 3:
        # single mask: (1, H, W)
        final_mask = masks[0] > 0.5
    else:
        # multi-mask: combine into one
        final_mask = combine_masks(masks.squeeze(1) > 0.5)

    return final_mask


def evaluate_misc():
    image_dir = misc_dir / "img"
    mask_dir = misc_dir / "mask"

    total_iou = 0.0
    total_dsc = 0.0
    total_sens = 0.0
    total_spec = 0.0
    total_images = 0

    for folder_name in ["f1", "f2", "f3", "f4", "f5"]:
        image_subfolder = image_dir / folder_name
        mask_subfolder = mask_dir / folder_name

        image_paths = list(image_subfolder.iterdir())
        with tqdm(image_paths, unit="img") as pbar:
            for image_path in pbar:
                image_name = image_path.stem
                pbar.set_description(f"Processing {image_name} in {folder_name}")
                mask_path = mask_subfolder / f"{image_name}.png"
                if not mask_path.is_file():
                    tqdm.write(f"[WARN] no label for {folder_name}-{image_name}")
                    logger.error(f"Mask not found: {mask_path}")
                    continue

                pil_image = Image.open(image_path).convert("RGB")
                # Apply rotation to JPEGs with EXIF orientation
                # (W, H)
                pil_image = ImageOps.exif_transpose(pil_image)
                # shape: (H, W, 3); dtype: uint8
                # image_np = np.array(pil_image)
                # H, W, _ = image_np.shape

                pil_mask = Image.open(mask_path).convert("L")
                # shape: (H, W), dtype: uint8
                mask_np = np.array(pil_mask)
                # dtype: bool
                true_mask = mask_np > 127

                # Each box: [x_min,y_min,x_max,y_max]
                boxes = mask_to_boxes(mask_np)
                # boxes = mask_to_boxes(mask_np, pad_mode="box")
                # shape: (num_contours,  num_points, 2); dtype: int
                # pos_points = mask_to_points_center(mask_np)
                # neg_points = sample_negative_points(mask_np, boxes)

                '''
                # Show prompts on mask
                mask_vis = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
                mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB)
                fig, ax = plt.subplots(1, figsize=(8, 6))
                ax.imshow(mask_vis)
                show_boxes(boxes, ax)
                show_points(pos_points, ax)
                # show_points(neg_points, ax)
                plt.axis("off")
                plt.show()
                '''

                pred_mask = fnet_inference(pil_image)
                # pred_mask = sam2_inference(image_np, boxes=boxes)
                # pred_mask = sam2_from_fnet(pil_image, boxes)
                ''''''
                if pred_mask.dtype != bool:
                    pred_mask = pred_mask > 0.5

                total_images += 1
                dsc = compute_dsc(pred_mask, true_mask)
                iou = compute_iou(pred_mask, true_mask)
                sens = compute_sens(pred_mask, true_mask)
                spec = compute_spec(pred_mask, true_mask)

                tqdm.write(
                    f"Image {folder_name}-{image_name}: IoU={iou:.4f}, Dice={dsc:.4f}, Sens={sens:.4f}, Spec={spec:.4f}"
                )

                total_dsc += dsc
                total_iou += iou
                total_sens += sens
                total_spec += spec

                logger.info(
                    f"Image {folder_name}-{image_name}: IoU={iou:.4f}, Dice={dsc:.4f}, Sens={sens:.4f}, Spec={spec:.4f}"
                )

                # fig, axs = plt.subplots(1, 2, figsize=(8, 6))
                # axs[0].imshow(true_mask, cmap="gray", vmin=0, vmax=1)
                # axs[1].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
                # axs[0].axis('off')
                # axs[1].axis('off')
                # plt.show()

                break
            break

    mean_iou = total_iou / total_images
    mean_dsc = total_dsc / total_images
    mean_sens = total_sens / total_images
    mean_spec = total_spec / total_images

    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean DSC: {mean_dsc:.4f}")
    print(f"Mean Sensitivity: {mean_sens:.4f}")
    print(f"Mean Specificity: {mean_spec:.4f}")

    logger.info(
        f"\n=== Summary over {total_images} images ===\n"
        f"Mean IoU  : {mean_iou:.4f}\n"
        f"Mean Dice : {mean_dsc:.4f}\n"
        f"Mean Sens : {mean_sens:.4f}\n"
        f"Mean Spec : {mean_spec:.4f}"
    )


def evaluate_fuseg():
    total_iou = 0.0
    total_dsc = 0.0
    total_sens = 0.0
    total_spec = 0.0
    total_images = 0

    test_dir = fuseg_dir / "test"
    image_dir = test_dir / "images"
    label_dir = test_dir / "labels"

    image_paths = [p for p in sorted(image_dir.iterdir()) if p.is_file()]
    with tqdm(image_paths, unit="img") as pbar:
        for image_path in pbar:
            image_filename = image_path.name
            pbar.set_description(f"Processing {image_filename}")

            label_path = label_dir / image_filename
            if not label_path.exists():
                tqdm.write(f"[WARN] no label for {image_filename}")
                logger.error(f"Mask not found: {image_filename}")
                continue

            pil_image = Image.open(image_path).convert("RGB")
            pil_mask = Image.open(label_path).convert("L")
            mask_np = np.array(pil_mask)
            true_mask = mask_np > 127

            boxes = mask_to_boxes(mask_np)

            '''
            # Show prompts on mask
            mask_vis = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
            mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB)
            fig, ax = plt.subplots(1, figsize=(8, 6))
            ax.imshow(mask_vis)
            show_boxes(boxes, ax)
            plt.axis("off")
            plt.show()
            '''

            # pred_mask = fnet_inference(pil_image)
            pred_mask = sam2_inference(pil_image, boxes=boxes)

            total_images += 1
            dsc = compute_dsc(pred_mask, true_mask)
            iou = compute_iou(pred_mask, true_mask)
            sens = compute_sens(pred_mask, true_mask)
            spec = compute_spec(pred_mask, true_mask)

            # tqdm.write(
            #     f"{image_filename}: IoU={iou:.4f}, Dice={dsc:.4f}, Sens={sens:.4f}, Spec={spec:.4f}"
            # )

            total_dsc += dsc
            total_iou += iou
            total_sens += sens
            total_spec += spec

            logger.info(
                f"{image_filename}: IoU={iou:.4f}, Dice={dsc:.4f}, Sens={sens:.4f}, Spec={spec:.4f}"
            )

            break

    mean_iou = total_iou / total_images
    mean_dsc = total_dsc / total_images
    mean_sens = total_sens / total_images
    mean_spec = total_spec / total_images

    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean DSC: {mean_dsc:.4f}")
    print(f"Mean Sensitivity: {mean_sens:.4f}")
    print(f"Mean Specificity: {mean_spec:.4f}")

    logger.info(
        f"\n=== Summary over {total_images} images ===\n"
        f"Mean IoU  : {mean_iou:.4f}\n"
        f"Mean Dice : {mean_dsc:.4f}\n"
        f"Mean Sens : {mean_sens:.4f}\n"
        f"Mean Spec : {mean_spec:.4f}"
    )


if __name__ == '__main__':
    # print(os.getcwd())
    logger = setup_logger(log_dir / "FUSeg" / "eval.log")
    # evaluate_misc()
    evaluate_fuseg()
    # results = collect_metrics("medsam")
    # for name, (mu, sd) in results.items():
    #     print(f"{name:4s}: {mu:6.2f} ± {sd:5.2f} %")
    # collect_negatives("sam2_neg-pt_box.log")

import re
from typing import Optional

import cv2
import numpy as np
import torch
import logging
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm
from pathlib import Path
from fnet import FNet
from PIL import Image, ImageOps, ImageDraw
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from segment_anything import sam_model_registry

if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

data_dir = Path.cwd().parents[0] / "Data"
test_dir = data_dir / "test"
image_dir = test_dir / "img"
mask_dir = test_dir / "mask"
neg_dir = test_dir / "neg"
log_dir = Path.cwd().parents[0] / "Results" / "eval"


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("Evaluation")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def compute_iou(pred_mask: np.ndarray,
                true_mask: np.ndarray,
                eps: float = 1e-6) -> float:
    """
    Compute IoU for two binary masks.

    Args:
        pred_mask (np.ndarray): Predicted mask, shape (H, W), values {0,1} or bool.
        true_mask (np.ndarray): Ground-truth mask, same shape as pred_mask.
        eps (float): Small constant to avoid division by zero.

    Returns:
        float: IoU score in [0,1].
    """
    # Ensure boolean arrays
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)

    # Intersection = pixels where both pred and true are 1
    intersection = np.logical_and(pred, true).sum()
    # Union = pixels where either pred or true is 1
    union = np.logical_or(pred, true).sum()

    return intersection / (union + eps)


def compute_dsc(pred_mask: np.ndarray,
                true_mask: np.ndarray,
                eps: float = 1e-6) -> float:
    """
    Compute Dice Similarity Coefficient (DSC) for two binary masks.

    Args:
        pred_mask (np.ndarray): Predicted mask, shape (H, W), values {0,1} or bool.
        true_mask (np.ndarray): Ground-truth mask, same shape as pred_mask.
        eps (float): Small constant to avoid division by zero.

    Returns:
        float: DSC score in [0,1].
    """
    # Convert to boolean
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)

    # Count intersection and sizes
    intersection = np.logical_and(pred, true).sum()
    size_pred = pred.sum()
    size_true = true.sum()

    # DSC formula
    return (2 * intersection + eps) / (size_pred + size_true + eps)


def compute_sens(pred_mask: np.ndarray,
                 true_mask: np.ndarray,
                 eps: float = 1e-6) -> float:
    """
    Compute Sensitivity (Recall, TPR) for two binary masks.

    Returns:
        float: Sensitivity in [0,1].
    """
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)

    # True Positives & False Negatives
    tp = np.logical_and(pred, true).sum()
    fn = np.logical_and(~pred, true).sum()

    return tp / (tp + fn + eps)


def compute_spec(pred_mask: np.ndarray,
                 true_mask: np.ndarray,
                 eps: float = 1e-6) -> float:
    """
    Compute Specificity (TNR) for two binary masks.

    Returns:
        float: Specificity in [0,1].
    """
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)

    # True Negatives & False Positives
    tn = np.logical_and(~pred, ~true).sum()
    fp = np.logical_and(pred, ~true).sum()

    return tn / (tn + fp + eps)


def mask_to_boxes(
        mask: np.ndarray,
        min_area: int = 0,
        pad_frac: float = 0.05,
        pad_mode: Optional[str] = None
) -> list[tuple[int, int, int, int]]:
    """
    Extracts bounding boxes around external contours in `mask`, then enlarges
    each box by a padding fraction.

    Args:
      mask:      2D or 3D uint8 array (H×W), foreground >127.
      min_area:  minimum area (in pixels) to keep a box.
      pad_frac:  fraction by which to enlarge each box.
      pad_mode:  "box" → pad relative to the box’s own width/height;
                 "image" → pad relative to the full image’s width/height.

    Returns:
      List of (x1, y1, x2, y2) tuples, with padding applied and
      clamped to image boundaries.
    """
    H, W = mask.shape[:2]
    _, bw = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area:
            continue

        if pad_mode == "box":
            pad_w = int(round(pad_frac * w))
            pad_h = int(round(pad_frac * h))
        elif pad_mode == "image":
            pad_w = int(round(pad_frac * W))
            pad_h = int(round(pad_frac * H))
        else:
            raise ValueError("pad_mode must be 'box' or 'image'")

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(W, x + w + pad_w)
        y2 = min(H, y + h + pad_h)

        boxes.append((x1, y1, x2, y2))

    return boxes


def jitter_box(box, image_size, alpha=0.01, mode="corner", clip=True):
    """
    Jitter a single bounding box by Gaussian noise proportional to its size.

    Args:
        box (array-like of 4 floats): [xmin, ymin, xmax, ymax] in pixel coords.
        image_size (tuple): (width, height) of the image for clamping.
        alpha (float): relative jitter scale (fraction of box width/height).
        mode (str): "corner" or "center_size" jitter strategy.
        clip (bool): whether to clamp output to [0, W]×[0, H].

    Returns:
        noisy_box (tuple of 4 floats): (xmin', ymin', xmax', ymax').
    """

    W_img, H_img = image_size

    x_min, y_min, x_max, y_max = box
    w = x_max - x_min
    h = y_max - y_min

    sigma_x = alpha * w
    sigma_y = alpha * h
    dx1, dy1 = np.random.normal(0, sigma_x), np.random.normal(0, sigma_y)
    dx2, dy2 = np.random.normal(0, sigma_x), np.random.normal(0, sigma_y)
    x_min = x_min + dx1
    y_min = y_min + dy1
    x_max = x_max + dx2
    y_max = y_max + dy2

    x_min, x_max = min(x_min, x_max), max(x_min, x_max)
    y_min, y_max = min(y_min, y_max), max(y_min, y_max)

    x_min = np.clip(x_min, 0, W_img)
    x_max = np.clip(x_max, 0, W_img)
    y_min = np.clip(y_min, 0, H_img)
    y_max = np.clip(y_max, 0, H_img)

    return x_min, y_min, x_max, y_max


def mask_to_points_center(
        mask: np.ndarray,
        num_points: int = 1,
        jitter_factor: float = 0.0
) -> list[list[tuple[int, int]]]:
    """
    For each connected foreground region in `mask`, compute the pixel
    at the center of the largest inscribed circle (the max-distance
    point in a distance-transform), then optionally jitter around it.
    Returns a list of length num_regions; each entry is a list of
    `num_points` (x, y) tuples for that region.
    """
    # --- 1) Binarize & label connected components ---
    _, bw = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels = cv2.connectedComponents(bw)

    results: list[list[tuple[int, int]]] = []

    # Skip label 0 (background)
    for lbl in range(1, num_labels):
        # build binary mask for this single component
        comp_mask = (labels == lbl).astype(np.uint8) * 255

        # --- 2) distance transform on that component mask ---
        dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)

        # find the “deepest” (max-distance) pixel
        flat_idx = np.argmax(dist)
        y0, x0 = np.unravel_index(flat_idx, dist.shape)
        r_max = dist[y0, x0]

        # --- 3) sample center + jitter inside that maximal circle ---
        pts_for_region: list[tuple[int, int]] = []
        for _ in range(num_points):
            if jitter_factor > 0 and r_max > 0:
                # sample radius so that (x0+dx, y0+dy) always stays within r_max
                jitter_r = jitter_factor * r_max
                # uniform in disk: radius ~ sqrt(U)*jitter_r
                r = np.sqrt(np.random.uniform(0, 1)) * jitter_r
                theta = np.random.uniform(0, 2 * np.pi)
                dx = int(r * np.cos(theta))
                dy = int(r * np.sin(theta))
                xi, yi = x0 + dx, y0 + dy
            else:
                xi, yi = x0, y0

            pts_for_region.append((xi, yi))

        results.append(pts_for_region)

    return results


def sample_negative_points(
        mask: np.ndarray,
        boxes: list[tuple[int, int, int, int]],
        num_neg_points: int = 4,
        min_dist_frac: float = 0.1
) -> list[list[tuple[int, int]]]:
    """
    For each bounding box in `boxes`, sample up to `num_neg_points` negative points
    inside the box but outside the true-mask area. Reject corners too close to the
    mask contour, then fall back on deepest background pixels.

    Args:
      mask:           2D uint8 array, 0=background, >0=foreground.
      boxes:          list of (x_min, y_min, x_max, y_max) tuples.
      num_neg_points: how many negatives per box (default 4).
      min_dist_frac:  minimum safe distance (as fraction of smaller box side)
                      for a corner to count.

    Returns:
      A list of lists; each sub-list is the negative-point coords for the
      corresponding box in `boxes`.
    """
    # Precompute global distance‐transform of background→foreground
    inv = (mask == 0).astype(np.uint8) * 255
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    H, W = dist.shape

    all_negs = []
    for (x0, y0, x1, y1) in boxes:
        w, h = x1 - x0, y1 - y0
        threshold = min(w, h) * min_dist_frac

        # Use x1-1, y1-1 so we don't step out of bounds
        corners = [
            (x0, y0),
            (x0, y1 - 1),
            (x1 - 1, y0),
            (x1 - 1, y1 - 1),
        ]

        # Filter out-of-bounds and those too close to the mask
        negs = []
        for x, y in corners:
            if 0 <= x < W and 0 <= y < H and dist[y, x] >= threshold:
                negs.append((x, y))
            if len(negs) == num_neg_points:
                break

        all_negs.append(negs)
    return all_negs


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


def fnet_inference(pil_image, image_size):
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

    H, W = image_size
    with torch.no_grad():
        out_logits, _ = model(img_tensor)
        pred = torch.sigmoid(out_logits)
        pred = F.interpolate(pred, size=(H, W), mode="area")
        pred = pred.squeeze().cpu().numpy()
        # print(pred.shape, pred.dtype)
        # predict = torch.round(torch.sigmoid(main_out)).byte()
        # pred_seg = predict.data.cpu().numpy() * 255

    return pred > 0.5


def medsam_inference(pil_image, boxes, image_size):
    ckpt_path = "ckpt/medsam_vit_b.pth"
    medsam_model = sam_model_registry["vit_b"](checkpoint=ckpt_path)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((1024, 1024), InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    # (3, H, W)
    image_tensor = preprocess(pil_image).to(device)
    # (B=1, 3, H, W)
    image_tensor = image_tensor.unsqueeze(0)

    H, W = image_size
    with torch.no_grad():
        image_embed = medsam_model.image_encoder(image_tensor)

        boxes_np = np.array(boxes)
        boxes_np = boxes_np / np.array([W, H, W, H]) * 1024
        boxes_tensor = torch.tensor(boxes_np, dtype=torch.float32, device=device)

        sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
            points=None,
            boxes=boxes_tensor,
            masks=None,
        )
        low_res_logits, _ = medsam_model.mask_decoder(
            image_embeddings=image_embed,  # (B, 256, 64, 64)
            image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

        low_res_pred = F.interpolate(
            low_res_pred,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )  # (1, 1, gt.shape)
        # (num_masks, H, W)
        low_res_pred = low_res_pred.squeeze().cpu().numpy()
        pred_masks = low_res_pred > 0.5

    if pred_masks.ndim == 2:
        return pred_masks
    else:
        return combine_masks(pred_masks)


def sam2_inference(
        image_np: np.ndarray,
        pos_points: list[list[tuple[int, int]]] = None,
        neg_points: list[list[tuple[int, int]]] = None,
        boxes: list[tuple[int, int, int, int]] = None
) -> Optional[np.ndarray]:
    sam2_checkpoint = "ckpt/sam2.1_hiera_base_plus.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image_np)

    tasks = []

    if pos_points:
        for pts in pos_points:
            coords = np.array(pts)
            labels = np.ones(len(pts), dtype=int)
            tasks.append((coords, labels, None))

    if boxes:
        if neg_points and len(neg_points) != len(boxes):
            raise ValueError("neg_points length must equal boxes length")
        for idx, box in enumerate(boxes):
            bg = neg_points[idx] if neg_points else []
            coords = np.array(bg) if bg else None
            labels = np.zeros(len(bg), dtype=int) if bg else None
            tasks.append((coords, labels, np.array(box)))

    if not tasks:
        logger.error("sam2_inference: must provide pos_points or boxes")
        return None

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


def evaluate():
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
                mask_path = mask_dir / folder_name / f"{image_name}.png"
                if not mask_path.is_file():
                    logger.error(f"Mask not found: {mask_path}")
                    continue

                pil_image = Image.open(image_path).convert("RGB")
                # Apply rotation to JPEGs with EXIF orientation
                pil_image = ImageOps.exif_transpose(pil_image)
                # shape: (H, W, 3); dtype: uint8
                image_np = np.array(pil_image)
                H, W, _ = image_np.shape

                pil_mask = Image.open(mask_path).convert("L")
                # shape: (H, W), dtype: uint8
                mask_np = np.array(pil_mask)
                # dtype: bool
                true_mask = mask_np > 127

                # print(mask_np.dtype)
                # print(mask_np.min(), mask_np.max())
                # unique_vals = np.unique(mask_np)
                # unique_set = set(unique_vals.tolist())
                # print(unique_set)

                # Each box: [x_min,y_min,x_max,y_max]
                boxes = mask_to_boxes(mask_np, pad_mode="box")
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
                # show_points(pos_points, ax)
                # show_points(neg_points, ax)
                plt.axis("off")
                plt.show()
                '''

                # pred_mask = fnet_inference(pil_image, (H, W))
                # pred_mask = medsam_inference(pil_image, new_boxes, (H, W))
                pred_mask = sam2_inference(image_np, boxes=boxes)
                # pred_mask = sam2_inference(image_np, pos_points=pos_points)
                # pred_mask = sam2_inference(image_np, neg_points=neg_points, boxes=boxes)
                ''''''
                if pred_mask.dtype != bool:
                    pred_mask = pred_mask > 0.5

                total_images += 1
                dsc = compute_dsc(pred_mask, true_mask)
                iou = compute_iou(pred_mask, true_mask)
                sens = compute_sens(pred_mask, true_mask)
                spec = compute_spec(pred_mask, true_mask)

                # print(f"Image {folder_name}-{image_name}: IoU={iou:.4f}, Dice={dsc:.4f}, Sens={sens:.4f}, Spec={spec:.4f}")

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


def collect_metrics(log_prefix):
    pat_iou = re.compile(r"^Mean\s+IoU\s*:\s*(?P<iou>\d+\.\d+)", re.MULTILINE)
    pat_dice = re.compile(r"^Mean\s+Dice\s*:\s*(?P<dice>\d+\.\d+)", re.MULTILINE)
    pat_sens = re.compile(r"^Mean\s+Sens\s*:\s*(?P<sens>\d+\.\d+)", re.MULTILINE)
    pat_spec = re.compile(r"^Mean\s+Spec\s*:\s*(?P<spec>\d+\.\d+)", re.MULTILINE)

    ious, dices, senss, specs = [], [], [], []

    for log_path in log_dir.iterdir():
        if not (
                log_path.is_file()
                and log_path.name.startswith(log_prefix)
                and log_path.name.endswith(".log")
        ):
            continue

        text = log_path.read_text()

        # Extract summary values
        m_iou = pat_iou.search(text)
        m_dice = pat_dice.search(text)
        m_sens = pat_sens.search(text)
        m_spec = pat_spec.search(text)

        if not (m_iou and m_dice and m_sens and m_spec):
            raise ValueError(f"Could not find summary metrics in {log_path.name}")

        # Convert to float and scale to percent
        ious.append(float(m_iou.group("iou")) * 100)
        dices.append(float(m_dice.group("dice")) * 100)
        senss.append(float(m_sens.group("sens")) * 100)
        specs.append(float(m_spec.group("spec")) * 100)

    # print(len(ious), len(dices), len(senss), len(specs))

    def mean_std(vals):
        arr = np.array(vals, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=1))

    return {
        "IoU": mean_std(ious),
        "Dice": mean_std(dices),
        "Sens": mean_std(senss),
        "Spec": mean_std(specs),
    }


def collect_negatives(log_file):
    log_path = log_dir / log_file

    pattern = re.compile(r"Image\s+([^-:]+-[^-:]+):\s+IoU=([0-9]*\.?[0-9]+)")

    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue

            name, iou = m.groups()
            iou = float(iou)
            if iou < 0.8:
                print(name, iou)
                folder_name, image_name = name.split("-")
                image_path = image_dir / folder_name / f"{image_name}.jpg"
                mask_path = mask_dir / folder_name / f"{image_name}.png"

                true_mask = Image.open(mask_path).convert("L")
                # shape: (H, W), dtype: uint8
                mask_np = np.array(true_mask)
                boxes = mask_to_boxes(mask_np)
                neg_points = sample_negative_points(mask_np, boxes)

                pil_image = Image.open(image_path).convert("RGB")
                # Apply rotation to JPEGs with EXIF orientation
                pil_image = ImageOps.exif_transpose(pil_image)
                # shape: (H, W, 3); dtype: uint8
                image_np = np.array(pil_image)
                H, W, _ = image_np.shape

                fnet_mask = fnet_inference(pil_image, (H, W))
                fnet_mask = (fnet_mask.astype(np.uint8)) * 255
                fnet_mask = Image.fromarray(fnet_mask, mode="L")

                sam2_mask = sam2_inference(image_np, boxes=boxes)
                sam2_mask = (sam2_mask.astype(np.uint8)) * 255
                sam2_mask = Image.fromarray(sam2_mask, mode="L")

                draw = ImageDraw.Draw(pil_image)
                for (x1, y1, x2, y2) in boxes:
                    draw.rectangle([x1, y1, x2, y2], outline="green", width=2)

                for pts in neg_points:
                    for (x, y) in pts:
                        # small red circle of radius 3px
                        r = 3
                        draw.ellipse(
                            [(x - r, y - r), (x + r, y + r)],
                            fill="red"
                        )

                neg_dir.mkdir(parents=True, exist_ok=True)
                pil_image.save(neg_dir / f"{name}_box.jpg")
                sam2_mask.save(neg_dir / f"{name}_sam2.png")
                fnet_mask.save(neg_dir / f"{name}_fnet.png")
                true_mask.save(neg_dir / f"{name}_true.png")

                # plt.figure(figsize=(8, 6))
                # plt.imshow(pil_image)
                # plt.imshow(sam2_mask, cmap="gray", vmin=0, vmax=255)
                # plt.axis('off')
                # plt.show()

                break


if __name__ == '__main__':
    # print(os.getcwd())
    logger = setup_logger("eval.log")
    evaluate()
    # results = collect_metrics("medsam")
    # for name, (mu, sd) in results.items():
    #     print(f"{name:4s}: {mu:6.2f} ± {sd:5.2f} %")
    # collect_negatives("sam2_neg-pt_box.log")

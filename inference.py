import os
import cv2
import numpy as np
import torch
import logging
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
from model.fnet import FNet
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from segment_anything import sam_model_registry


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("Evaluation")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fh = logging.FileHandler(log_file)
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


def mask_to_boxes(mask, min_area=100):
    _, bw = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, x + w, y + h))

        # area = w * h
        # if area >= min_area:
        #     boxes.append((x, y, w, h))

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


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


def show_boxes(boxes, ax):
    for box in boxes:
        show_box(box, ax)


def fnet_inference(img_path, ckpt_path):
    model = FNet()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    # RGB
    img = Image.open(img_path)

    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),  # → FloatTensor in [0,1], shape C×H×W
    ])

    # (1, C, 512, 512)
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        out_logits, _ = model(img_tensor)
        print(out_logits.shape)
        # predict = torch.round(torch.sigmoid(main_out)).byte()
        # pred_seg = predict.data.cpu().numpy() * 255

    # result = Image.fromarray(pred_seg.squeeze(), mode='L')


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

    return combine_masks(pred_masks)


def sam2_inference(image_np, boxes=None):
    boxes_np = np.array(boxes)

    sam2_checkpoint = "ckpt/sam2.1_hiera_base_plus.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image_np)

    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes_np,
        multimask_output=False,
    )
    masks = masks > 0.5
    if masks.ndim == 3:
        assert masks.shape[0] == 1
        return combine_masks(masks)
    else:
        assert masks.shape[1] == 1
        return combine_masks(masks.squeeze(1))


if __name__ == '__main__':
    # print(os.getcwd())
    logger = setup_logger("eval.log")

    total_iou = 0.0
    total_dsc = 0.0
    total_sens = 0.0
    total_spec = 0.0
    total_images = 0

    data_dir = Path.cwd().parents[0] / "Data"
    test_dir = data_dir / "test"
    image_dir = test_dir / "img"
    mask_dir = test_dir / "mask"

    for folder_name in ["f1", "f2", "f3", "f4", "f5"]:
        image_subfolder = image_dir / folder_name
        mask_subfolder = mask_dir / folder_name

        image_paths = list(image_subfolder.iterdir())
        for image_path in tqdm(image_paths, desc=f"Images in {folder_name}", unit="img"):
            img_name = image_path.stem
            mask_path = mask_dir / folder_name / f"{img_name}.png"
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
            boxes = mask_to_boxes(mask_np)
            new_boxes = []
            for box in boxes:
                new_box = jitter_box(box, (W, H))
                new_boxes.append(new_box)

            # fig, ax = plt.subplots(1)
            # ax.imshow(image_np)
            # show_boxes(new_boxes, ax)
            # plt.show()


            # fnet_inference(img_path, ckpt_path)
            # pred_mask = medsam_inference(pil_image, new_boxes, (H, W))
            pred_mask = sam2_inference(image_np, new_boxes)
            ''''''
            if pred_mask.dtype != bool:
                pred_mask = pred_mask > 0.5

            total_images += 1
            dsc = compute_dsc(pred_mask, true_mask)
            iou = compute_iou(pred_mask, true_mask)
            sens = compute_sens(pred_mask, true_mask)
            spec = compute_spec(pred_mask, true_mask)

            total_dsc += dsc
            total_iou += iou
            total_sens += sens
            total_spec += spec

            logger.info(
                f"Image {folder_name}-{img_name}: IoU={iou:.4f}, Dice={dsc:.4f}, Sens={sens:.4f}, Spec={spec:.4f}"
            )

            # fig, axs = plt.subplots(1, 2)
            # axs[0].imshow(true_mask, cmap="gray", vmin=0, vmax=1)
            # axs[1].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
            # plt.show()

            # break
        # break

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



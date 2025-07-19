import cv2
import numpy as np

from typing import Optional


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
            pad_w = pad_h = 0

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
from PIL import Image
import numpy as np
import cv2

def overlay_mask_edge(
    pil_image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),  # lime-green for medical overlays
    thickness: int = 2
) -> Image.Image:
    """
    Draws the external contour of a boolean mask onto a PIL RGB image.

    Args:
      pil_image: PIL.Image in RGB mode.
      mask:      H×W boolean or uint8 array (True/1 = foreground).
      color:     (R, G, B) color for the edge (default is lime-green).
      thickness: line thickness in pixels.

    Returns:
      A new PIL.Image with the mask edge overlaid.
    """
    # 1) Convert PIL.Image to numpy RGB array
    rgb = np.array(pil_image.convert("RGB"))

    # 2) Ensure mask is uint8 binary (0 or 255)
    bin_mask = (mask.astype(bool)).astype(np.uint8) * 255

    # 3) Find external contours with full fidelity
    contours, _ = cv2.findContours(
        bin_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    # 4) Convert RGB -> BGR for OpenCV drawing
    overlay_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # 5) Draw contours in BGR order
    color_bgr = (color[2], color[1], color[0])
    cv2.drawContours(
        overlay_bgr,
        contours,
        contourIdx=-1,  # draw all
        color=color_bgr,
        thickness=thickness,
        lineType=cv2.LINE_AA
    )

    # 6) Convert back to RGB and return as PIL.Image
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(overlay_rgb)

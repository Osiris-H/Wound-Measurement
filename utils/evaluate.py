import numpy as np


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

    # Special case: no foreground in either mask
    if union == 0:
        return 1.0

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

    # Count foreground pixels in each
    size_pred = pred.sum()
    size_true = true.sum()

    # Special case: no foreground in either mask
    if size_pred == 0 and size_true == 0:
        return 1.0

    # Compute intersection
    intersection = np.logical_and(pred, true).sum()


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

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


def mask_to_points_contour(
        mask: np.ndarray,
        num_points: int = 8
) -> list[tuple[int, int]]:
    """
    Given a 2D uint8 mask, find all external closed contours
    and return `num_points_per_contour` roughly-uniformly-spaced points
    along each one. The result will be an array of shape
    (num_contours * num_points_per_contour, 2).
    """
    # 1) Binarize
    _, bw = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    # 2) Find external contours (no approximation so we get full boundary)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contours found in mask")

    sampled_pts = []
    # 3) For each contour, sample exactly num_points_per_contour points
    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        # close the loop for length calc
        pts_closed = np.vstack([pts, pts[0]])
        segs = np.diff(pts_closed, axis=0)
        lengths = np.hypot(segs[:, 0], segs[:, 1])
        cumlen = np.cumsum(lengths)
        total_len = cumlen[-1]
        if total_len == 0:
            # degenerate contour: repeat the single point
            sampled_pts.extend([pts[0]] * num_points)
            continue

        # distances at which to sample
        dists = np.linspace(0, total_len, num_points, endpoint=False)
        for d in dists:
            idx = np.searchsorted(cumlen, d)
            sampled_pts.append(pts_closed[idx])

    return sampled_pts


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

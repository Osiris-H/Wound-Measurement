


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


def mask_to_points_center(mask: np.ndarray,
                          num_points: int = 1,
                          jitter_factor: float = 0.0) -> list[tuple[int, int]]:
    """
    For each external contour in `mask`, compute the pixel
    at the maximum distance from the contour boundary (i.e.
    center of the largest inscribed circle), then optionally
    jitter around it.

    Args:
      mask: 2D uint8 array, nonzero = foreground.
      num_points: how many points to return per contour.
      jitter_factor: fraction of the max-distance to use as radius
                     for uniform random jitter (0 = no jitter).

    Returns:
      A list of (x, y) integer tuples inside each contour,
      length = num_contours * num_points_per_contour.
    """
    # 1) Extract all external contours
    _, bw = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts: list[tuple[int, int]] = []

    for cnt in contours:
        # build a mask only for this contour
        single_mask = np.zeros_like(mask)
        cv2.drawContours(single_mask, [cnt], -1, color=255, thickness=-1)

        # distance transform: for each pixel, dist to nearest zero
        dist = cv2.distanceTransform(single_mask, cv2.DIST_L2, 5)
        # find the “deepest” point
        y0, x0 = np.unravel_index(np.argmax(dist), dist.shape)
        r_max = dist[y0, x0]

        for _ in range(num_points):
            if jitter_factor > 0 and r_max > 0:
                theta = np.random.uniform(0, 2 * np.pi)
                r = np.random.uniform(0, jitter_factor * r_max)
                dx = int(r * np.cos(theta))
                dy = int(r * np.sin(theta))
                x, y = x0 + dx, y0 + dy
                h, w = mask.shape
                x = max(0, min(w - 1, x))
                y = max(0, min(h - 1, y))
                if single_mask[y, x] == 0:
                    x, y = x0, y0
            else:
                x, y = x0, y0

            pts.append((x, y))

    return pts


def sample_negative_points(
        mask: np.ndarray,
        boxes: list[tuple[int, int, int, int]],
        num_neg_points: int = 4,
        min_dist_frac: float = 0.1
) -> list[tuple[int, int]]:
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

    all_negs = []

    for (x0, y0, x1, y1) in boxes:
        w, h = x1 - x0, y1 - y0
        min_dist = min(w, h) * min_dist_frac

        # 1) try the four corners directly
        corners = [(x0, y0), (x0, y1), (x1, y0), (x1, y1)]
        negs = []
        for (x, y) in corners:
            if dist[y, x] >= min_dist:
                negs.append((x, y))
            if len(negs) == num_neg_points:
                break

        # 2) fallback: pick the deepest background pixels in the box
        if len(negs) < num_neg_points:
            needed = num_neg_points - len(negs)
            sub = dist[y0:y1 + 1, x0:x1 + 1]
            flat = sub.flatten()
            idxs = np.argsort(-flat)  # descending distances
            added = 0
            for idx in idxs:
                if added >= needed:
                    break
                yy, xx = divmod(idx, sub.shape[1])
                xi, yi = x0 + xx, y0 + yy
                if mask[yi, xi] != 0 or (xi, yi) in negs:
                    continue
                negs.append((xi, yi))
                added += 1

        # all_negs.extend(negs[:num_neg_points])
        all_negs.append(negs[:num_neg_points])

    return all_negs
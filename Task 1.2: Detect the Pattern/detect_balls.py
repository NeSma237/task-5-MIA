"""
Task 1.2 - Detect the Pattern (IMPROVED VERSION)
Classical computer vision system (OpenCV only) to detect red and blue balls
in images and generate YOLO-format label files.

Class IDs:
    0 = Blue
    1 = Red

Changes vs. the first version (aimed at raising recall / TP without letting
false positives explode):
  1. Wider HSV ranges for both colors (catches balls in shadow / low light,
     where saturation and value drop).
  2. Relaxed shape filters (circularity, aspect ratio, extent, min area) so
     partially-shadowed or slightly-occluded balls aren't thrown away.
  3. Smaller morphological closing kernel, so two DIFFERENT nearby balls are
     less likely to be fused into a single blob.
  4. NEW: distance-transform + watershed splitting step. If a blob still
     represents two touching/overlapping balls of the same color, this
     splits it into separate detections instead of counting it as one
     (or rejecting it entirely for failing the circularity test).

Author: Nesma
"""

import cv2
import numpy as np
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_DIR = r"C:\Users\nesma\Downloads\Task_1_2_Submission\balls\balls"
OUTPUT_DIR = r"C:\Users\nesma\Downloads\Task_1_2_Submission\labels"
TARGET_WIDTH = 800

# HSV color ranges. Red S/V minimums raised back up specifically because
# warm-toned shadows and neutral/brownish surfaces (cloth, pavement in
# shade) were passing through the looser thresholds and being detected as
# false "red" balls.
COLOR_RANGES = {
    1: [  # Red
        (np.array([0,   110, 55]), np.array([10,  255, 255])),
        (np.array([170, 110, 55]), np.array([180, 255, 255])),
    ],
    0: [  # Blue
        (np.array([100, 70, 35]), np.array([130, 255, 255])),
    ],
}

# Filtering thresholds. Back to the original (stricter) circularity - the
# opening-recovery experiment showed that loosening this let through too
# many false positives elsewhere, and wasn't a net win.
MIN_AREA_RATIO   = 0.00035
MIN_CIRCULARITY  = 0.70
MIN_ASPECT_RATIO = 0.65
MAX_ASPECT_RATIO = 1.35
MIN_EXTENT       = 0.50

# Watershed splitting is now only attempted on a blob that (a) fails the
# normal single-ball validation AND (b) is clearly larger than one ball
# would be - i.e. it looks like it's probably two merged balls, not noise.
# This ratio is "how many times bigger than the minimum ball area" a blob
# has to be before we even try to split it.
SPLIT_AREA_MULTIPLIER = 2.5

# Fraction of a connected component's max distance-transform value used to
# seed "sure foreground" markers for watershed. Lower = splits more
# aggressively.
WATERSHED_PEAK_RATIO = 0.5

# Two boxes of the SAME class with IoU above this are considered
# duplicates of the same ball; only the larger one is kept.
NMS_IOU_THRESHOLD = 0.3

# Gate for attempting opening-based recovery: the ORIGINAL (unopened) blob
# must already be at least this circular. A true ball merged with a thin
# bridge to clutter is still roughly round overall; chaotic background
# texture (wood grain, shelving, etc.) that happens to pass the color
# threshold is not, and would otherwise cough up random small circular
# fragments once opened with several kernel sizes.
RECOVERY_MIN_ORIGINAL_CIRCULARITY = 0.40

# After opening, the recovered contour must retain at least this fraction
# of the original blob's area - otherwise we're likely just grabbing an
# unrelated small round fragment out of a large noisy blob, not the real
# ball with a thin bridge trimmed off.
RECOVERY_MIN_AREA_FRACTION = 0.45


# ---------------------------------------------------------------------------
# Step 1: Load and prepare the image
# ---------------------------------------------------------------------------
def load_and_prepare(img_path, target_width=TARGET_WIDTH):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    scale = target_width / w
    img = cv2.resize(img, (target_width, int(h * scale)))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_ch = clahe.apply(v_ch)

    hsv = cv2.merge([h_ch, s_ch, v_ch])
    return img, hsv


# ---------------------------------------------------------------------------
# Step 2: Build a clean binary mask for one color
# ---------------------------------------------------------------------------
def build_color_mask(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

    # Smaller opening kernel: still kills thin noise/wires without eating
    # away at small distant balls.
    kernel_open = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # Closing kernel: merges a ball's own color panels back together. A
    # square kernel here (confirmed empirically to work better than an
    # elliptical one on this dataset - the elliptical version merged
    # background clutter less consistently, causing it to fragment into
    # several smaller blobs that individually slipped past the shape
    # filters more often).
    kernel_close = np.ones((18, 18), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    return mask


# ---------------------------------------------------------------------------
# Step 3a: Validate a single contour as a plausible ball
# ---------------------------------------------------------------------------
def _circularity(cnt):
    """Convex-hull-based circularity: 1.0 = perfect circle. Returns 0.0
    for degenerate contours (zero hull perimeter)."""
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)
    if hull_perimeter == 0:
        return 0.0
    return 4 * np.pi * hull_area / (hull_perimeter ** 2)


def _validate_contour(cnt, img_area):
    area = cv2.contourArea(cnt)
    if area < img_area * MIN_AREA_RATIO:
        return None

    circularity = _circularity(cnt)
    if circularity < MIN_CIRCULARITY:
        return None

    x, y, bw, bh = cv2.boundingRect(cnt)
    if bh == 0:
        return None

    aspect_ratio = bw / bh
    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        return None

    extent = area / (bw * bh)
    if extent < MIN_EXTENT:
        return None

    return (x, y, bw, bh)


# ---------------------------------------------------------------------------
# Step 3b: Recover a ball that's merged with a thin bridge of noise
# ---------------------------------------------------------------------------
def _recover_via_opening(cnt, mask_shape, img_area):
    """
    Handles the case where a real ball's blob is otherwise round and valid,
    but is welded to a separate nearby object (e.g. a similarly-colored
    tank, cable, or wall edge) by a thin strip of touching pixels - which
    ruins its circularity/extent and gets the whole blob rejected.

    A morphological OPENING (erosion then dilation) with a growing kernel
    breaks thin connecting strips while leaving large round blobs mostly
    intact, since erosion removes anything thinner than the kernel long
    before it eats into a wide, solid blob. Once the bridge is gone, the
    ball's own contour should validate on its own.
    """
    original_area = cv2.contourArea(cnt)
    if original_area <= 0:
        return []

    # Don't even try on blobs that are already shapeless clutter - only a
    # blob that's reasonably close to round to begin with is a candidate
    # for "real ball with a thin bridge attached", not random background
    # texture that happened to pass the color threshold.
    if _circularity(cnt) < RECOVERY_MIN_ORIGINAL_CIRCULARITY:
        return []

    local_mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.drawContours(local_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    for k in (5, 7, 9, 11, 15):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        opened = cv2.morphologyEx(local_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        if cv2.countNonZero(opened) == 0:
            break

        sub_contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not sub_contours:
            continue

        largest = max(sub_contours, key=cv2.contourArea)

        # Reject fragments that only account for a small slice of the
        # original blob - that's a random chunk of clutter, not "the same
        # ball with a bridge trimmed off".
        if cv2.contourArea(largest) < original_area * RECOVERY_MIN_AREA_FRACTION:
            continue

        box = _validate_contour(largest, img_area)
        if box:
            return [box]

    return []


# ---------------------------------------------------------------------------
# Step 3c: Split a blob that looks like two merged balls (watershed)
# ---------------------------------------------------------------------------
def _split_blob(cnt, mask_shape, img_area):
    """
    Only called on a blob that already FAILED whole-shape validation and is
    clearly bigger than one ball. Isolates that single blob into its own
    small mask and uses a distance-transform + watershed pass to try to
    separate it into 2+ individual balls. Returns a list of validated
    (x, y, w, h) boxes (possibly empty if nothing splits out cleanly).
    """
    local_mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.drawContours(local_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    dist = cv2.distanceTransform(local_mask, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return []

    _, sure_fg = cv2.threshold(dist, WATERSHED_PEAK_RATIO * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    if num_markers <= 2:
        # Distance transform didn't find more than one peak -> this blob
        # really is just one (oddly-shaped) object, not two merged balls.
        # Don't force a split.
        return []

    markers = markers + 1
    unknown = cv2.subtract(local_mask, sure_fg)
    markers[unknown == 255] = 0

    mask_bgr = cv2.cvtColor(local_mask, cv2.COLOR_GRAY2BGR)
    cv2.watershed(mask_bgr, markers)

    boxes = []
    for label in range(2, num_markers + 1):
        blob_mask = np.uint8(markers == label) * 255
        if cv2.countNonZero(blob_mask) == 0:
            continue
        sub_contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for sub_cnt in sub_contours:
            box = _validate_contour(sub_cnt, img_area)
            if box:
                boxes.append(box)
    return boxes


# ---------------------------------------------------------------------------
# Step 3c: Remove duplicate/overlapping boxes (same ball detected twice)
# ---------------------------------------------------------------------------
def _non_max_suppression(boxes, iou_thresh=NMS_IOU_THRESHOLD):
    if not boxes:
        return boxes

    def box_iou(a, b):
        ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
        bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    boxes_sorted = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for b in boxes_sorted:
        if all(box_iou(b, k) <= iou_thresh for k in kept):
            kept.append(b)
    return kept


# ---------------------------------------------------------------------------
# Step 3d: Turn a cleaned mask into validated ball detections
# ---------------------------------------------------------------------------
def find_balls_in_mask(mask, img_area):
    """
    Returns a list of (x, y, w, h) bounding boxes in pixel coordinates.

    Strategy: try to validate each blob AS A WHOLE first (the common,
    correct case: one blob = one ball). Only if that fails, AND the blob
    is clearly bigger than a single ball should be, do we attempt to split
    it into multiple balls via watershed. This avoids fragmenting normal,
    valid single balls into several false positives.
    """
    boxes = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * MIN_AREA_RATIO:
            continue

        box = _validate_contour(cnt, img_area)
        if box is not None:
            boxes.append(box)
            continue

        # Whole blob failed as a single ball. Only try splitting it if
        # it's clearly larger than a normal single ball (probably two
        # real balls merged together, not just noise).
        if area >= img_area * MIN_AREA_RATIO * SPLIT_AREA_MULTIPLIER:
            boxes.extend(_split_blob(cnt, mask.shape, img_area))

    return _non_max_suppression(boxes)


# ---------------------------------------------------------------------------
# Step 4: Full detection pipeline for one image
# ---------------------------------------------------------------------------
def detect_balls(img_path):
    img, hsv = load_and_prepare(img_path)
    img_area = img.shape[0] * img.shape[1]

    detections = []
    for class_id, ranges in COLOR_RANGES.items():
        mask = build_color_mask(hsv, ranges)
        boxes = find_balls_in_mask(mask, img_area)
        for (x, y, bw, bh) in boxes:
            detections.append((class_id, x, y, bw, bh))

    return img, detections


# ---------------------------------------------------------------------------
# Step 5: Convert pixel bounding boxes to normalized YOLO format
# ---------------------------------------------------------------------------
def to_yolo_format(class_id, x, y, bw, bh, img_w, img_h):
    x_center = (x + bw / 2) / img_w
    y_center = (y + bh / 2) / img_h
    width_norm = bw / img_w
    height_norm = bh / img_h
    return class_id, x_center, y_center, width_norm, height_norm


# ---------------------------------------------------------------------------
# Step 6: Run the pipeline on every image and write the label files
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_files = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    for fname in image_files:
        img_path = os.path.join(INPUT_DIR, fname)
        img, detections = detect_balls(img_path)
        img_h, img_w = img.shape[:2]

        txt_name = os.path.splitext(fname)[0] + '.txt'
        txt_path = os.path.join(OUTPUT_DIR, txt_name)

        with open(txt_path, 'w') as f:
            for class_id, x, y, bw, bh in detections:
                cid, xc, yc, wn, hn = to_yolo_format(class_id, x, y, bw, bh, img_w, img_h)
                f.write(f"{cid} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

        print(f"{fname}: {len(detections)} ball(s) detected -> {txt_name}")


if __name__ == "__main__":
    main()

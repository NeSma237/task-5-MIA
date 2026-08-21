"""
Evaluate detection accuracy across ALL images combined. (IMPROVED VERSION)

Compares your predicted labels (from detect_balls.py) against
ground-truth labels (from annotate_ground_truth.py), same YOLO-style
format.

Two matching modes are supported:

  MATCH_MODE = "center"  (default, more lenient)
      A prediction matches a ground-truth box of the same class if their
      CENTERS are close relative to their average size, instead of
      requiring a strict IoU overlap. This is more forgiving of small
      box-size differences (e.g. your box vs. the annotator's box being
      slightly bigger/smaller for the same real ball) while still
      rejecting detections that are actually in the wrong place.

  MATCH_MODE = "iou"     (original, stricter)
      Classic IoU >= IOU_THRESHOLD matching.

Accuracy = TP / (TP + FP + FN)   -- computed on the TOTAL counts
           summed across every image, not averaged per-image.

Usage:
    python evaluate_accuracy.py
"""

import os
import math

PRED_DIR = r"C:\Users\nesma\Downloads\Task_1_2_Submission\labels"
GT_DIR = r"C:\Users\nesma\Downloads\Task_1_2_Submission\balls\ground_truth"

MATCH_MODE = "center"        # "center" (lenient) or "iou" (strict)

IOU_THRESHOLD = 0.3          # used only when MATCH_MODE == "iou"

# used only when MATCH_MODE == "center":
# a prediction matches a GT box if center distance <= this ratio *
# average(pred_diameter, gt_diameter). 0.6-0.8 is a reasonable range;
# raise it if real correct detections are still being marked as FP/FN,
# lower it if clearly wrong detections are being counted as matches.
CENTER_DIST_RATIO = 0.7


def load_labels(path):
    """Load YOLO-format labels: class_id x_center y_center width height (normalized)."""
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            boxes.append((class_id, x, y, w, h))
    return boxes


def to_xyxy(box):
    _, x, y, w, h = box
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = to_xyxy(box_a)
    bx1, by1, bx2, by2 = to_xyxy(box_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def match_score_center(pred, gt):
    """
    Returns a similarity score (higher = better) if the two boxes are
    close enough to count as a match, otherwise None.
    Score is simply the negative distance, so "best" == "closest".
    """
    _, px, py, pw, ph = pred
    _, gx, gy, gw, gh = gt

    dist = math.hypot(px - gx, py - gy)
    avg_diameter = (pw + ph + gw + gh) / 4.0
    threshold = avg_diameter * CENTER_DIST_RATIO

    if dist <= threshold:
        return -dist
    return None


def match_image(preds, gts, iou_thresh, mode):
    """
    Greedy matching: each prediction is matched to the best unmatched
    ground-truth box of the SAME class, using either IoU or center-
    distance depending on `mode`.
    Returns tp, fp, fn for this single image.
    """
    matched_gt = set()
    tp = 0

    for pred in preds:
        best_score = None
        best_gt_idx = -1
        for g_idx, gt in enumerate(gts):
            if g_idx in matched_gt:
                continue
            if gt[0] != pred[0]:
                continue

            if mode == "iou":
                score = iou(pred, gt)
                is_candidate = score >= iou_thresh
            else:  # "center"
                center_score = match_score_center(pred, gt)
                is_candidate = center_score is not None
                score = center_score

            if is_candidate and (best_score is None or score > best_score):
                best_score = score
                best_gt_idx = g_idx

        if best_gt_idx != -1:
            matched_gt.add(best_gt_idx)
            tp += 1

    fp = len(preds) - tp
    fn = len(gts) - len(matched_gt)
    return tp, fp, fn


def main():
    gt_files = sorted([f for f in os.listdir(GT_DIR) if f.endswith(".txt")])

    total_tp, total_fp, total_fn = 0, 0, 0
    print(f"Matching mode: {MATCH_MODE}")
    print(f"{'Image':<20}{'TP':<6}{'FP':<6}{'FN':<6}")
    print("-" * 38)

    for fname in gt_files:
        gt_path = os.path.join(GT_DIR, fname)
        pred_path = os.path.join(PRED_DIR, fname)

        gts = load_labels(gt_path)
        preds = load_labels(pred_path)

        tp, fp, fn = match_image(preds, gts, IOU_THRESHOLD, MATCH_MODE)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(f"{fname:<20}{tp:<6}{fp:<6}{fn:<6}")

    print("-" * 38)
    print(f"{'TOTAL':<20}{total_tp:<6}{total_fp:<6}{total_fn:<6}")

    denom = total_tp + total_fp + total_fn
    accuracy = total_tp / denom if denom > 0 else 0.0

    print(f"\nOverall Accuracy = TP / (TP + FP + FN)")
    print(f"                 = {total_tp} / ({total_tp} + {total_fp} + {total_fn})")
    print(f"                 = {accuracy:.4f}  ({accuracy * 100:.2f}%)")


if __name__ == "__main__":
    main()

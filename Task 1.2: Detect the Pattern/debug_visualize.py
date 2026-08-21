"""
Debug visualization for detect_balls.py

For every image, saves:
  debug/<name>_boxes.png  -> original image with detected boxes drawn
  debug/<name>_mask_red.png   -> cleaned red mask
  debug/<name>_mask_blue.png  -> cleaned blue mask

Run this AFTER detect_balls.py (it reuses the same functions/config),
then open the debug/ folder and check the images where you have
FP or FN (e.g. ball_8, ball_9) to see exactly what went wrong.
"""

import os
import cv2
import numpy as np

# import everything from your existing detection script
from detect_balls import (
    INPUT_DIR, load_and_prepare, build_color_mask, find_balls_in_mask,
    COLOR_RANGES
)

DEBUG_DIR = "debug"
os.makedirs(DEBUG_DIR, exist_ok=True)

COLOR_DRAW = {0: (255, 0, 0), 1: (0, 0, 255)}   # BGR: blue box, red box
COLOR_NAME = {0: "blue", 1: "red"}

image_files = sorted(f for f in os.listdir(INPUT_DIR)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png')))

for fname in image_files:
    img_path = os.path.join(INPUT_DIR, fname)
    img, hsv = load_and_prepare(img_path)
    img_area = img.shape[0] * img.shape[1]
    vis = img.copy()

    base = os.path.splitext(fname)[0]

    for class_id, ranges in COLOR_RANGES.items():
        mask = build_color_mask(hsv, ranges)
        boxes = find_balls_in_mask(mask, img_area)

        # save the mask so you can see exactly what the color threshold caught
        mask_path = os.path.join(DEBUG_DIR, f"{base}_mask_{COLOR_NAME[class_id]}.png")
        cv2.imwrite(mask_path, mask)

        # draw every accepted box on the visualization image
        for (x, y, bw, bh) in boxes:
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), COLOR_DRAW[class_id], 2)
            cv2.putText(vis, COLOR_NAME[class_id], (x, max(0, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_DRAW[class_id], 2)

    boxes_path = os.path.join(DEBUG_DIR, f"{base}_boxes.png")
    cv2.imwrite(boxes_path, vis)
    print(f"{fname}: saved debug images -> {base}_boxes.png, "
          f"{base}_mask_red.png, {base}_mask_blue.png")

print(f"\nAll debug images saved in: {DEBUG_DIR}")
print("Open the _boxes.png for images with FP/FN to see what went wrong,")
print("and check the matching _mask_red.png / _mask_blue.png to see the raw color detection.")

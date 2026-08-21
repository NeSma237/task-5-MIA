"""
Ground-truth annotation tool (IMPROVED VERSION).

The first version used a FIXED box size for every ball, which caused a
mismatch against real (variable-size) predicted boxes and hurt IoU-based
accuracy even when detections were essentially correct. This version
captures the REAL size of each ball with two clicks:

  1st click: the CENTER of the ball.
             LEFT click  -> Blue ball  (class_id = 0)
             RIGHT click -> Red ball   (class_id = 1)
  2nd click: any point on the EDGE of the same ball (roughly its border).
             This is used to compute the ball's radius -> real box size.

Repeat for every ball you see, then close the window (or press 'n') to
move to the next image. Middle-click undoes the last completed ball.

Output: a "ground_truth" folder with one .txt file per image, same format
as your predictions:
    <class_id> <x_center> <y_center> <width> <height>   (normalized 0-1)
"""

import os
import math
import cv2
import matplotlib.pyplot as plt

IMAGES_DIR = r"C:\Users\nesma\Downloads\Task_1_2_Submission\balls\balls"
OUTPUT_DIR = r"C:\Users\nesma\Downloads\Task_1_2_Submission\ground_truth"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# each entry: (class_id, x_center_px, y_center_px, diameter_px)
clicks = []
pending = {"class_id": None, "cx": None, "cy": None, "marker": None}


def onclick(event):
    if event.xdata is None or event.ydata is None:
        return
    x, y = event.xdata, event.ydata

    # Middle click = undo last finished ball
    if event.button == 2:
        if clicks:
            clicks.pop()
            plt.gca().lines[-1].remove() if plt.gca().lines else None
            plt.draw()
        return

    if pending["class_id"] is None:
        # First click of a new ball: determines color + center
        if event.button == 1:
            pending["class_id"] = 0
            color = 'b'
        elif event.button == 3:
            pending["class_id"] = 1
            color = 'r'
        else:
            return
        pending["cx"], pending["cy"] = x, y
        plt.plot(x, y, marker='+', color=color, markersize=12, mew=2)
        plt.draw()
    else:
        # Second click: edge point -> compute radius, finish this ball
        dx = x - pending["cx"]
        dy = y - pending["cy"]
        radius = math.hypot(dx, dy)
        radius = max(radius, 2)  # avoid degenerate zero-size boxes

        color = 'b' if pending["class_id"] == 0 else 'r'
        circle = plt.Circle((pending["cx"], pending["cy"]), radius,
                             color=color, fill=False, linewidth=2)
        plt.gca().add_patch(circle)
        plt.draw()

        clicks.append((pending["class_id"], pending["cx"], pending["cy"], radius * 2))
        pending["class_id"] = None
        pending["cx"] = None
        pending["cy"] = None


image_files = sorted([f for f in os.listdir(IMAGES_DIR)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

print(f"Found {len(image_files)} images. For each ball:")
print("  1) LEFT click on its CENTER = Blue, RIGHT click on its CENTER = Red")
print("  2) click again on its EDGE to set the real size")
print("  Middle click = undo last ball. Close the window when done with an image.\n")

for img_name in image_files:
    img_path = os.path.join(IMAGES_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read {img_name}, skipping.")
        continue
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    clicks.clear()
    pending["class_id"] = None
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_rgb)
    ax.set_title(f"{img_name}  |  click center then edge  |  L=Blue R=Red  |  middle=undo")
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    base_name = os.path.splitext(img_name)[0]
    out_path = os.path.join(OUTPUT_DIR, base_name + ".txt")
    with open(out_path, "w") as f:
        for class_id, cx_px, cy_px, diameter_px in clicks:
            x_norm = cx_px / w
            y_norm = cy_px / h
            w_norm = diameter_px / w
            h_norm = diameter_px / h
            f.write(f"{class_id} {x_norm:.6f} {y_norm:.6f} "
                     f"{w_norm:.6f} {h_norm:.6f}\n")

    print(f"Saved {len(clicks)} ground-truth balls -> {out_path}")

print("\nDone! Ground truth saved in:", OUTPUT_DIR)
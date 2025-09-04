import os
import cv2
import numpy as np
from typing import List, Dict

def _color_for_group(group_id: str):
    color_palette = {
        "G1": (144, 238, 144),  # Light Green
        "G2": (0, 128, 0),      # Dark Green  
        "G3": (255, 182, 193),  # Light Pink
        "G4": (220, 20, 60),    # Crimson Red
        "G5": (173, 216, 230),  # Light Blue
        "G6": (0, 0, 139),      # Dark Blue
        "G7": (255, 255, 224),  # Light Yellow
        "G8": (255, 140, 0),    # Dark Orange
        "G9": (221, 160, 221),  # Plum Purple
        "G10": (128, 0, 128),   # Purple
        "G11": (255, 192, 203), # Pink
        "G12": (165, 42, 42),   # Brown
        "G13": (64, 224, 208),  # Turquoise
        "G14": (255, 20, 147),  # Deep Pink
        "G15": (50, 205, 50),   # Lime Green
    }
    if group_id in color_palette:
        return color_palette[group_id]
    else:
        np.random.seed(abs(hash(group_id)) % (2**32 - 1))
        hue = np.random.randint(0, 180)
        saturation = np.random.randint(150, 255)
        value = np.random.randint(150, 255)
        hsv_color = np.uint8([[[hue, saturation, value]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        return (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))

def draw_grouped_detections(image_path: str, detections: List[Dict], save_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        gid = det.get("group_id", "G0")
        score = det.get("score", 0.0)
        label = f"{gid} {score:.2f}"
        color = _color_for_group(gid)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - h - baseline), (x1 + w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)

"""Step 2 - skin patches (regions of interest) and their mean colour per frame.

Each region is a polygon through MediaPipe landmark indices. "left"/"right" are the
person's own left/right (so "cheek_left" appears on the right side of the image).
"""
import cv2
import numpy as np

REGIONS = {
    # Lower forehead (between the eyebrows and the hairline/fringe). Forehead was the
    # strongest region in Verkruysse 2008. Split in two halves so the detector can compare them.
    "forehead_right": [151, 108, 69, 66, 107, 9],
    "forehead_left": [151, 337, 299, 296, 336, 9],
    "cheek_left": [266, 426, 436, 416, 376, 352, 347, 330],
    "cheek_right": [36, 206, 216, 192, 147, 123, 118, 101],
    "chin": [201, 200, 421, 428, 396, 175, 171, 208],
}

# Start with forehead + cheeks (the doc's advice); the detector compares all five.
DEFAULT_REGIONS = ["forehead_right", "forehead_left", "cheek_left", "cheek_right"]
DETECTOR_REGIONS = list(REGIONS)


def region_polygon(landmarks, name):
    return landmarks[REGIONS[name]].astype(np.int32)


def region_mean_rgb(rgb, landmarks, name, min_pixels=20, trim=(15, 10)):
    """Mean (R, G, B) of the pixels inside one region. NaN if the patch is too small/off-frame."""
    poly = region_polygon(landmarks, name)
    h, w = rgb.shape[:2]
    x0, y0 = np.maximum(poly.min(axis=0), 0)
    x1, y1 = np.minimum(poly.max(axis=0) + 1, [w, h])
    if x1 - x0 < 2 or y1 - y0 < 2:
        return np.full(3, np.nan)
    crop = rgb[y0:y1, x0:x1]
    mask = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly - [x0, y0]], 1)
    m = mask.astype(bool)
    if m.sum() < min_pixels:
        return np.full(3, np.nan)
    px = crop[m].reshape(-1, 3).astype(np.float32)
    if trim and len(px) >= 4 * min_pixels:
        # Drop non-skin pixels: the darkest (glasses frames, beard, shadow) and the
        # brightest (shine, reflections). Keep the middle of the brightness range.
        lum = px @ np.array([0.299, 0.587, 0.114], np.float32)
        lo, hi = np.percentile(lum, [trim[0], 100 - trim[1]])
        keep = (lum >= lo) & (lum <= hi)
        if keep.sum() >= min_pixels:
            px = px[keep]
    return px.mean(axis=0)


def head_pose(landmarks):
    """Per-frame head position/size/turn from landmarks, used to detect and cancel motion.
    Returns [nose_x, nose_y, face_width, yaw, pitch]; yaw/pitch are ~0 when facing the camera."""
    l, r, top, chin, nose = landmarks[234], landmarks[454], landmarks[10], landmarks[152], landmarks[1]
    width = float(np.hypot(*(r - l)))
    height = float(np.hypot(*(chin - top))) + 1e-6
    yaw = float(np.dot(nose - l, r - l) / (width ** 2 + 1e-6) - 0.5)
    pitch = float(np.dot(nose - top, chin - top) / height ** 2 - 0.5)
    return np.array([nose[0], nose[1], width, yaw, pitch], dtype=np.float64)


def draw_regions(rgb, landmarks, names=None, draw_points=True):
    """Return a copy of the frame with landmarks and region outlines drawn (for checking)."""
    out = rgb.copy()
    if draw_points:
        for x, y in landmarks.astype(int):
            cv2.circle(out, (int(x), int(y)), 1, (0, 255, 255), -1)
    colours = [(255, 60, 60), (60, 255, 60), (60, 120, 255), (255, 200, 0), (255, 0, 255), (0, 255, 255)]
    for i, name in enumerate(names or REGIONS):
        poly = region_polygon(landmarks, name)
        cv2.polylines(out, [poly], True, colours[i % len(colours)], 2)
        cx, cy = poly.mean(axis=0).astype(int)
        cv2.putText(out, name, (int(cx) - 30, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colours[i % len(colours)], 1)
    return out

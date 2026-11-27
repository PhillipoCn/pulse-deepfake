"""Step 1 - read video frames and find the face (OpenCV + MediaPipe Face Landmarker)."""
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


def video_fps(path) -> float:
    """The real frame rate. Every beats-per-minute number depends on it.
    A <video>.fps.json sidecar (written by record.py with the measured rate) wins."""
    side = Path(str(path) + ".fps.json")
    if side.exists():
        import json
        return float(json.load(open(side))["fps"])
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 1 or fps > 240:
        raise ValueError(f"Bad frame rate {fps} reported for {path}")
    return float(fps)


def iter_frames(path, max_frames=None, start_frame=0):
    """Yield frames as RGB uint8 arrays."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    n = 0
    try:
        while max_frames is None or n < max_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            n += 1
    finally:
        cap.release()


class FaceTracker:
    """MediaPipe Face Landmarker: 478 points per face, returned in pixel coordinates."""

    def __init__(self, video_mode=True):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing model file {MODEL_PATH}")
        self.video_mode = video_mode
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE,
            num_faces=1,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)

    def landmarks(self, rgb, timestamp_ms=0):
        """Return a (478, 2) float array of (x, y) pixels, or None if no face."""
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self.video_mode:
            res = self._lm.detect_for_video(image, int(timestamp_ms))
        else:
            res = self._lm.detect(image)
        if not res.face_landmarks:
            return None
        h, w = rgb.shape[:2]
        return np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]], dtype=np.float32)

    def close(self):
        self._lm.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

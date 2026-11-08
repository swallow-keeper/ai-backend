"""Local MediaPipe/OpenCV feature extraction matching the mobile modules.

Raw frames are never saved by this module. Input videos and human annotations
stay on the research machine. A model task file must already exist locally.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from swallow_ml.temporal import FEATURE_VERSION


class MotionExtractor:
    def __init__(self, model: Path):
        import cv2
        import mediapipe as mp
        self.cv = cv2
        self.mp = mp
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1, min_face_detection_confidence=.6, min_face_presence_confidence=.6,
        )
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self.previous = None
        self.last_time = 0

    def reset(self):
        self.previous = None
        self.last_time = 0

    def close(self):
        self.detector.close()
        self.reset()

    def _flow(self, gray, box):
        x, y, w, h = box
        points = np.array([(xx, yy) for yy in range(y + 2, y + h - 2, 4)
                           for xx in range(x + 2, x + w - 2, 4)], dtype=np.float32).reshape(-1, 1, 2)
        if len(points) < 3:
            return np.zeros(4)
        after, status, error = self.cv.calcOpticalFlowPyrLK(
            self.previous, gray, points, None, winSize=(15, 15), maxLevel=2,
            criteria=(self.cv.TERM_CRITERIA_COUNT | self.cv.TERM_CRITERIA_EPS, 20, .03),
        )
        valid = status.ravel().astype(bool) & (error.ravel() < 30) & np.isfinite(after).all(axis=(1, 2))
        if valid.sum() < 3:
            return np.zeros(4)
        delta = (after - points)[valid, 0, :]
        return np.array([*delta.mean(axis=0), np.linalg.norm(delta, axis=1).mean(), valid.mean()])

    def push(self, bgr, timestamp: float):
        cv = self.cv
        height = int(np.floor(bgr.shape[0] * 320 / bgr.shape[1] + .5))
        image = cv.resize(bgr, (320, height))
        rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        faces = self.detector.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)).face_landmarks
        if not faces:
            self.reset(); return None
        face = faces[0]
        left = min(face[234].x, face[454].x) * 320
        right = max(face[234].x, face[454].x) * 320
        top = face[10].y * height; chin = face[152].y * height
        fw = right - left; fh = chin - top; center = (left + right) / 2
        if fw < 32 or fh < 32 or top < 1 or chin + fh * .6 >= height or left < 1 or right >= 319 or abs(face[1].x * 320 - center) / fw > .22:
            self.reset(); return None
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        dt = (timestamp - self.last_time) / 1000
        if self.previous is None or self.previous.shape != gray.shape or dt <= 0 or dt > .4:
            self.previous = gray; self.last_time = timestamp; return None

        def box(x0, y0, x1, y1):
            return max(1, int(np.floor(x0))), max(1, int(np.floor(y0))), max(2, int(np.floor(x1 - x0))), max(2, int(np.floor(y1 - y0)))

        head = self._flow(gray, box(left + fw * .15, top, right - fw * .15, top + fh * .35))
        neck_top = chin - fh * .04; neck_height = fh * .64
        bands = [self._flow(gray, box(center - fw * .28, neck_top + neck_height * i / 3,
                                     center + fw * .28, neck_top + neck_height * (i + 1) / 3)) for i in range(3)]
        if head[3] < .3 or any(band[3] < .3 for band in bands):
            self.reset(); return None
        x, y, w, h = box(left + fw * .25, top + fh * .6, right - fw * .25, top + fh * .9)
        mouth = cv.absdiff(self.previous[y:y+h, x:x+w], gray[y:y+h, x:x+w]).mean() / 255
        scale = 1 / (fw * dt)
        features = [-(band[1] - head[1]) * scale for band in bands] + [bands[0][2] * scale, mouth, -head[1] * scale]
        self.previous = gray; self.last_time = timestamp
        return np.clip(features, -10, 10).tolist()


def extract_dataset(manifest: Path, model: Path, output: Path):
    import cv2
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    extractor = MotionExtractor(model)
    count = 0
    try:
        with output.open("w", encoding="utf-8") as stream:
            for entry in entries:
                video = (manifest.parent / entry["video"]).resolve()
                events = np.asarray(entry["swallows_ms"], dtype=np.float64)
                # An annotation is a candidate's endpoint, not the start of neck movement.
                if not entry.get("subject_id") or events.ndim != 1 or not np.isfinite(events).all() or (events < 0).any():
                    raise ValueError("subject_id and finite swallow endpoint timestamps are required")
                capture = cv2.VideoCapture(str(video))
                if not capture.isOpened():
                    raise ValueError(f"Cannot open local video: {video.name}")
                fps = capture.get(cv2.CAP_PROP_FPS)
                if not np.isfinite(fps) or fps < 8:
                    capture.release(); raise ValueError("Input video must have a known frame rate of at least 8 fps")
                extractor.reset(); history = []; frame_index = 0; next_time = 0
                try:
                    while True:
                        ok, frame = capture.read()
                        if not ok: break
                        timestamp = frame_index * 1000 / fps; frame_index += 1
                        if timestamp + .001 < next_time: continue
                        next_time += 125
                        features = extractor.push(frame, timestamp)
                        if features is None:
                            history = []; continue
                        history.append((timestamp, features)); history = history[-16:]
                        if len(history) < 16: continue
                        age = timestamp - events
                        label = int(np.any((age >= 0) & (age <= 250)))
                        # Drop ambiguous near-event negatives; do not label an entire swallow as no-swallow.
                        if not label and events.size and np.min(np.abs(age)) < 1000: continue
                        row = {"subject_id": str(entry["subject_id"]), "source": video.name, "end_ms": timestamp,
                               "feature_version": FEATURE_VERSION, "label": label, "features": [value for _, value in history]}
                        stream.write(json.dumps(row, allow_nan=False) + "\n"); count += 1
                finally:
                    capture.release()
    finally:
        extractor.close()
    if not count:
        raise ValueError("No usable windows extracted; check face/neck visibility and annotations")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--face-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Extracted {extract_dataset(args.manifest, args.face_model, args.output)} labeled windows")


if __name__ == "__main__":
    main()

"""Synthetic image geometry checks, not swallow/clinical performance tests."""
from types import SimpleNamespace

import numpy as np
import pytest

cv = pytest.importorskip("cv2")
from swallow_ml.motion import MotionExtractor


def test_lk_recovers_translation_and_rejects_untextured_regions():
    rng = np.random.default_rng(8)
    before = cv.GaussianBlur(rng.integers(0, 256, (160, 160), dtype=np.uint8), (3, 3), 0)
    after = cv.warpAffine(before, np.float32([[1, 0, 2], [0, 1, -1]]), (160, 160))
    extractor = MotionExtractor.__new__(MotionExtractor)
    extractor.cv = cv; extractor.previous = before
    result = extractor._flow(after, (20, 20, 100, 100))
    assert result[0] == pytest.approx(2, abs=.1)
    assert result[1] == pytest.approx(-1, abs=.1)
    assert result[3] > .9
    extractor.previous = np.zeros_like(before)
    assert extractor._flow(np.zeros_like(before), (20, 20, 100, 100))[3] == 0


def test_head_translation_is_removed_from_neck_motion():
    face = [SimpleNamespace(x=.5, y=.4) for _ in range(478)]
    for index, x, y in [(10, .5, .15), (152, .5, .55), (234, .25, .4), (454, .75, .4), (1, .5, .3)]:
        face[index] = SimpleNamespace(x=x, y=y)
    extractor = MotionExtractor.__new__(MotionExtractor)
    extractor.cv = cv
    extractor.mp = SimpleNamespace(Image=lambda **kwargs: kwargs, ImageFormat=SimpleNamespace(SRGB=1))
    extractor.detector = SimpleNamespace(detect=lambda image: SimpleNamespace(face_landmarks=[face]))
    extractor.reset()
    rng = np.random.default_rng(9)
    image = cv.GaussianBlur(rng.integers(0, 256, (480, 320, 3), dtype=np.uint8), (3, 3), 0)
    shifted = cv.warpAffine(image, np.float32([[1, 0, 0], [0, 1, -2]]), (320, 480))
    assert extractor.push(image, 125) is None
    result = extractor.push(shifted, 250)
    assert result is not None
    assert np.abs(result[:3]).max() < .01
    assert result[5] == pytest.approx(.1, abs=.01)

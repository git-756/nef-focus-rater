# -*- coding: utf-8 -*-
"""core のテスト。exiftool がない環境ではスキップ。

ピント判定は合成画像で、NEF 関連は実ファイル（tests/data/*.NEF があれば）で確認する。
"""
import os
import shutil

import cv2
import numpy as np
import pytest

from nef_focus_rater import core


def _scene(seed=1, h=2000, w=3000):
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 120, np.uint8)
    for _ in range(400):
        x, y, r = int(rng.integers(0, w)), int(rng.integers(0, h)), int(rng.integers(10, 80))
        cv2.circle(img, (x, y), r, int(rng.integers(0, 255)), -1)
        cv2.rectangle(img, (x, y), (x + r, y + r // 2), int(rng.integers(0, 255)), 3)
    return img


def _noisy(img, sigma=6, seed=2):
    rng = np.random.default_rng(seed)
    out = np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)
    return cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 92])[1], cv2.IMREAD_GRAYSCALE)


def _score(gray):
    lap = core._laplacian(gray)
    return core.global_score(lap, core.noise_floor(lap))[0]


def test_sharp_beats_blurred_even_with_noise():
    base = _scene()
    sharp = _score(_noisy(base))
    mild = _score(_noisy(cv2.GaussianBlur(base, (0, 0), 3)))
    heavy = _score(_noisy(cv2.GaussianBlur(base, (0, 0), 9)))
    assert sharp > mild * 5 > heavy


def test_background_bokeh_is_not_blur():
    base = _scene()
    bokeh = base.copy()
    bokeh[:, :1800] = cv2.GaussianBlur(base, (0, 0), 25)[:, :1800]
    mild = _score(_noisy(cv2.GaussianBlur(base, (0, 0), 3)))
    assert _score(_noisy(bokeh)) > mild * 3


def test_af_area_score():
    base = _scene()
    img = base.copy()
    img[:, :1800] = cv2.GaussianBlur(base, (0, 0), 25)[:, :1800]
    lap = core._laplacian(_noisy(img))
    floor = core.noise_floor(lap)
    sharp_side = core.af_score(lap, (0.85, 0.5, 0.1, 0.1), floor)
    blur_side = core.af_score(lap, (0.2, 0.5, 0.1, 0.1), floor)
    assert sharp_side > blur_side * 3


def test_jpeg_size():
    data = cv2.imencode(".jpg", np.zeros((123, 456, 3), np.uint8))[1].tobytes()
    assert core.jpeg_size(data) == (456, 123)
    assert core.jpeg_size(b"not a jpeg") is None


needs_exiftool = pytest.mark.skipif(core.find_exiftool() is None, reason="exiftool がありません")
DATA = os.path.join(os.path.dirname(__file__), "data")
SAMPLES = sorted(f for f in os.listdir(DATA) if f.lower().endswith(".nef")) if os.path.isdir(DATA) else []


@needs_exiftool
@pytest.mark.skipif(not SAMPLES, reason="tests/data に NEF がありません")
def test_real_nef_roundtrip(tmp_path):
    """実際の NEF をコピーして、解析とレーティングの書き込み・読み戻しを確認する。"""
    sub = tmp_path / "日本語フォルダ"
    sub.mkdir()
    src = os.path.join(DATA, SAMPLES[0])
    dst = str(sub / "写真.NEF")
    shutil.copy2(src, dst)
    with core.ExifTool() as et:
        ph = core.read_metadata(et, [dst])[0]
        core.analyze(ph, None, et)
        assert ph.error == ""
        assert ph.score_global is not None
        before = os.path.getmtime(dst)
        s = core.write_ratings(et, [dst], 1, keep_backup=False, only_unrated=False)
        assert s.updated == 1
        assert core.read_ratings(et, [dst])[os.path.normpath(dst)] == 1
        assert abs(os.path.getmtime(dst) - before) < 1  # -P で更新日時を維持
        assert not os.path.exists(dst + "_original")
        # ★済みはスキップされる
        s = core.write_ratings(et, [dst], 2, only_unrated=True)
        assert s.skipped == 1
        assert core.read_ratings(et, [dst])[os.path.normpath(dst)] == 1

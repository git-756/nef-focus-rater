# -*- coding: utf-8 -*-
"""core のテスト。exiftool がない環境ではスキップ。

ピント判定は合成画像で、NEF 関連は実ファイル（tests/data/*.NEF があれば）で確認する。
"""
import base64
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


def _xmp_packet(rating_tag="<xmp:Rating>0</xmp:Rating>", pad=2000):
    """NX Studio が調整ファイルに埋め込む XMP（末尾は空白のパディング）。"""
    body = ('<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?><x:xmpmeta xmlns:x="adobe:ns:meta/">'
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            '<rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/">'
            '<xmp:CreatorTool>NIKON Z6_3 Ver.02.00   </xmp:CreatorTool>%s </rdf:Description>'
            '</rdf:RDF></x:xmpmeta>' % rating_tag)
    return (body + " " * pad + '<?xpacket end="w"?>').encode("utf-8")


def _nksc_bytes(rating=0, packet=None, nine=True):
    flat = base64.b64encode(packet if packet is not None else _xmp_packet("<xmp:Rating>%d</xmp:Rating>" % rating))
    b64 = b"&#xA;".join(flat[i:i + 76] for i in range(0, len(flat), 76))
    return ('<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
            '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 5.5.0">\n'
            ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
            '  <rdf:Description rdf:about="" xmlns:ast="http://ns.nikon.com/asteroid/1.0/"'
            ' xmlns:nine="http://ns.nikon.com/nine/1.0/">\n'
            '   <ast:XMLPackets rdf:parseType="Resource">\n'
            '    <rdf:value>').encode() + b64 + ('</rdf:value>\n'
            '   </ast:XMLPackets>\n'
            '   <nine:NineEdits>&lt;userData&gt;編集内容&lt;/userData&gt;</nine:NineEdits>\n'
            '   <nine:Label>0</nine:Label>\n'
            + ('   <nine:Rating>%d</nine:Rating>\n' % rating if nine else '') +
            '  </rdf:Description>\n </rdf:RDF>\n</x:xmpmeta>\n<?xpacket end="w"?>').encode("utf-8")


def _make_nksc(folder, name, data):
    d = folder / "NKSC_PARAM"
    d.mkdir(exist_ok=True)
    p = d / (name + ".nksc")
    p.write_bytes(data)
    return str(p)


def _embedded_xmp(n):
    return core._nksc_packet(open(n, "rb").read())[1]


def test_nksc_rating_roundtrip(tmp_path):
    nef = tmp_path / "写真.NEF"
    nef.write_bytes(b"")
    assert core.nksc_path(str(nef)) is None
    n = _make_nksc(tmp_path, "写真.NEF", _nksc_bytes(0))
    assert core.nksc_path(str(nef)) == n
    assert core.read_nksc_rating(n) == 0
    before = open(n, "rb").read()
    assert core.write_nksc_rating(n, 3) == "updated"
    after = open(n, "rb").read()
    assert core.read_nksc_rating(n) == 3
    assert b"<xmp:Rating>3</xmp:Rating>" in _embedded_xmp(n)   # NX Studio が表示に使う★
    assert b"<nine:Rating>3</nine:Rating>" in after
    # ★の部分以外は変わらない（長さも同じ）
    assert after == _nksc_bytes(3) and len(before) == len(after)
    assert core.write_nksc_rating(n, 3) == "unchanged"
    assert not os.path.exists(n + "_original")
    assert core.write_nksc_rating(n, 1, keep_backup=True) == "updated"
    assert core.read_nksc_rating(n + "_original") == 3
    assert not os.path.exists(n + ".tmp")


def test_nksc_inserts_rating_into_padding(tmp_path):
    """埋め込み XMP に★の項目がない（古い機種など）場合は、パディングを使って差し込む。"""
    packet = _xmp_packet("")
    n = _make_nksc(tmp_path, "a.NEF", _nksc_bytes(packet=packet, nine=False))
    before = open(n, "rb").read()
    assert core.read_nksc_rating(n) == 0
    assert core.write_nksc_rating(n, 2) == "updated"
    xmp = _embedded_xmp(n)
    assert len(xmp) == len(packet) and len(open(n, "rb").read()) == len(before)
    assert b"<xmp:Rating>2</xmp:Rating></rdf:Description>" in xmp
    assert xmp.endswith(b'<?xpacket end="w"?>')
    assert core.read_nksc_rating(n) == 2


def test_nksc_unsupported_is_untouched(tmp_path):
    # パディングが足りない／埋め込み XMP がない場合は書き換えない
    for name, data in (("a.NEF", _nksc_bytes(packet=_xmp_packet("", pad=5))),
                       ("b.NEF", b'<x:xmpmeta><nine:Rating>0</nine:Rating></x:xmpmeta>')):
        n = _make_nksc(tmp_path, name, data)
        assert core.write_nksc_rating(n, 1) == "notag"
        assert open(n, "rb").read() == data


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


@needs_exiftool
@pytest.mark.skipif(not SAMPLES, reason="tests/data に NEF がありません")
def test_real_nef_with_nksc(tmp_path):
    """NX Studio の調整ファイルがある NEF では、本体と調整ファイルの両方に★を書き込む。"""
    dst = str(tmp_path / "写真.NEF")
    shutil.copy2(os.path.join(DATA, SAMPLES[0]), dst)
    n = _make_nksc(tmp_path, "写真.NEF", _nksc_bytes(0))
    with core.ExifTool() as et:
        core.write_ratings(et, [dst], 0, only_unrated=False, nksc=False)  # 本体の★を 0 にそろえる
        s = core.write_ratings(et, [dst], 1, only_unrated=True)
        assert (s.updated, s.nksc_updated, s.nksc_failed) == (1, 1, 0)
        assert core.read_nksc_rating(n) == 1
        assert core.read_ratings(et, [dst], nksc=False)[os.path.normpath(dst)] == 1
        # 調整ファイルだけに★があっても「★済み」としてスキップする
        core.write_ratings(et, [dst], 0, only_unrated=False, nksc=False)
        core.write_nksc_rating(n, 4)
        s = core.write_ratings(et, [dst], 1, only_unrated=True)
        assert s.skipped == 1 and s.nksc_updated == 0
        assert core.read_ratings(et, [dst])[os.path.normpath(dst)] == 4
        assert core.read_ratings(et, [dst], nksc=False)[os.path.normpath(dst)] == 0
        # nksc=False なら調整ファイルには触れない
        s = core.write_ratings(et, [dst], 2, only_unrated=False, nksc=False)
        assert core.read_nksc_rating(n) == 4

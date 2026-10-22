# -*- coding: utf-8 -*-
"""Qt に依存しない中核処理：exiftool 連携、プレビュー読み込み、ピント判定、レーティング書き込み。"""
from __future__ import annotations

import base64
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

RAW_EXTS = (".nef", ".nrw")
JPEG_EXTS = (".jpg", ".jpeg")
GRID_COLS, GRID_ROWS = 8, 6   # 画像全体を評価するときのブロック分割
TOP_RATIO = 0.15              # シャープなブロック上位◯%の平均を「全体スコア」にする
NOISE_FLOOR_PCT = 2           # ノイズフロア推定に使う細分ブロックの下位パーセンタイル
META_CHUNK = 100              # メタデータを一括取得するときの1回あたりの枚数
WRITE_CHUNK = 50
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# 解析解像度（長辺ピクセル。None はプレビューのフル解像度）
RESOLUTIONS = {"full": None, "half": 3000, "quarter": 1500}


# --------------------------------------------------------------------------
# exiftool
# --------------------------------------------------------------------------
def find_exiftool() -> Optional[List[str]]:
    """exiftool の実行コマンドを探す。

    優先順: 環境変数 NEF_FOCUS_RATER_EXIFTOOL → このパッケージの隣 → PATH →
    macOS の Homebrew 既定パス（Finder から起動すると PATH に入っていないため）。
    """
    env = os.environ.get("NEF_FOCUS_RATER_EXIFTOOL")
    if env:
        return shlex.split(env, posix=(os.name != "nt"))
    roots = [Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent]
    if getattr(sys, "frozen", False):
        roots.insert(0, Path(sys.executable).resolve().parent)
    for root in roots:
        for name in ("exiftool.exe", "exiftool"):
            p = root / name
            if p.is_file():
                return [str(p)]
    found = shutil.which("exiftool")
    if found:
        return [found]
    for p in ("/opt/homebrew/bin/exiftool", "/usr/local/bin/exiftool", "/usr/bin/exiftool"):
        if os.path.isfile(p):
            return [p]
    return None


class ExifToolError(RuntimeError):
    pass


class ExifTool:
    """exiftool を -stay_open モードで常駐させて使う（起動コストが1回で済む）。

    引数は標準入力から UTF-8 で渡すので、日本語のフォルダ名・ファイル名でも安全。
    スレッドセーフ（内部でロック）。
    """

    def __init__(self, cmd: Optional[Sequence[str]] = None):
        cmd = list(cmd or find_exiftool() or [])
        if not cmd:
            raise ExifToolError(
                "exiftool が見つかりません。インストールして PATH に通すか、"
                "環境変数 NEF_FOCUS_RATER_EXIFTOOL で場所を指定してください。")
        self._p = subprocess.Popen(
            cmd + ["-stay_open", "True", "-@", "-", "-common_args", "-charset", "filename=utf8"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=NOWIN)
        self._lock = threading.Lock()
        self._n = 0
        self._err = bytearray()
        self._err_cv = threading.Condition()
        threading.Thread(target=self._pump_stderr, daemon=True).start()

    def _pump_stderr(self):
        while True:
            chunk = self._p.stderr.read1(65536)
            if not chunk:
                break
            with self._err_cv:
                self._err += chunk
                self._err_cv.notify_all()

    def execute(self, args: Iterable[str]) -> Tuple[bytes, str]:
        with self._lock:
            if self._p.poll() is not None:
                raise ExifToolError("exiftool が終了しています")
            self._n += 1
            marker = ("{ready%d}" % self._n).encode()
            lines = [str(a) for a in args] + ["-echo4", "{ready%d}" % self._n, "-execute%d" % self._n]
            self._p.stdin.write(("\n".join(lines) + "\n").encode("utf-8"))
            self._p.stdin.flush()
            out = bytearray()
            while True:
                chunk = self._p.stdout.read1(1 << 20)
                if not chunk:
                    raise ExifToolError("exiftool の出力が途中で途切れました")
                out += chunk
                k = out.rfind(marker)
                if k != -1 and not out[k + len(marker):].strip():
                    del out[k:]
                    break
            with self._err_cv:
                while marker not in self._err:
                    if not self._err_cv.wait(timeout=30) and self._p.poll() is not None:
                        break
                k = self._err.find(marker)
                if k == -1:
                    err = bytes(self._err)
                    self._err.clear()
                else:
                    err = bytes(self._err[:k])
                    end = self._err.find(b"\n", k)
                    del self._err[:(end + 1) if end != -1 else len(self._err)]
            return bytes(out), err.decode("utf-8", "replace").strip()

    def close(self):
        try:
            if self._p.poll() is None:
                self._p.stdin.write(b"-stay_open\nFalse\n")
                self._p.stdin.flush()
                self._p.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self._p.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------
# ファイル列挙・メタデータ
# --------------------------------------------------------------------------
def list_raw_files(folder: str, recursive: bool = True) -> List[str]:
    out = []
    walker = os.walk(folder) if recursive else [(folder, [], os.listdir(folder))]
    for root, dirs, files in walker:
        # NX Studio の調整ファイルフォルダやバックアップは除外
        dirs[:] = [d for d in dirs if d not in ("NKSC_PARAM", "_Original") and not d.startswith(".")]
        for f in files:
            if f.lower().endswith(RAW_EXTS) and not f.startswith("._"):
                out.append(os.path.join(root, f))
    return sorted(out)


def sibling_jpeg(path: str) -> Optional[str]:
    """RAW+JPEG 同時記録の JPEG（同じフォルダ・同じファイル名）を探す。"""
    stem = os.path.splitext(path)[0]
    for ext in (".JPG", ".jpg", ".JPEG", ".jpeg"):
        p = stem + ext
        if os.path.isfile(p):
            return p
    return None


PREVIEW_TAGS = ("JpgFromRaw", "PreviewImage", "OtherImage")
META_TAGS = (
    ["-Model", "-Orientation#", "-XMP:Rating", "-FocusResult", "-AFAreaMode",
     "-AFImageWidth", "-AFImageHeight", "-AFAreaXPosition", "-AFAreaYPosition",
     "-AFAreaWidth", "-AFAreaHeight"]
    + ["-%sStart" % t for t in PREVIEW_TAGS] + ["-%sLength" % t for t in PREVIEW_TAGS]
)


@dataclass
class Photo:
    path: str
    model: str = ""
    orientation: int = 1
    rating: Optional[int] = None
    focus_result: str = ""          # カメラが記録した AF 結果（Focus / Out of Focus）
    af_mode: str = ""
    af_box: Optional[Tuple[float, float, float, float]] = None  # (cx, cy, w, h) 0〜1 の正規化座標
    previews: List[Tuple[str, int, int]] = field(default_factory=list)  # (タグ名, 開始, 長さ)
    preview_tag: str = ""
    preview_size: Optional[Tuple[int, int]] = None  # (幅, 高さ)
    score_global: Optional[float] = None
    score_af: Optional[float] = None
    best_block: Optional[Tuple[float, float]] = None  # 最もシャープなブロックの中心（正規化）
    error: str = ""

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    def score(self, prefer_af: bool = True) -> Optional[float]:
        if prefer_af and self.score_af is not None:
            return self.score_af
        return self.score_global


def _to_int(v) -> Optional[int]:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def read_metadata(et: ExifTool, paths: Sequence[str],
                  progress: Optional[Callable[[int], None]] = None) -> List[Photo]:
    """プレビュー位置・AFエリア・レーティングなどを、exiftool の一括呼び出しで取得する。"""
    photos: dict = {}
    done = 0
    for i in range(0, len(paths), META_CHUNK):
        chunk = list(paths[i:i + META_CHUNK])
        out, _err = et.execute(["-json", "-fast", "-api", "LargeFileSupport=1"] + META_TAGS + chunk)
        try:
            items = json.loads(out.decode("utf-8", "replace") or "[]")
        except json.JSONDecodeError:
            items = []
        for d in items:
            src = os.path.normpath(d.get("SourceFile", ""))
            ph = Photo(path=src)
            ph.model = str(d.get("Model", ""))
            ph.orientation = _to_int(d.get("Orientation")) or 1
            r = d.get("Rating")
            ph.rating = _to_int(r) if r is not None else 0
            nr = read_nksc_rating(nksc_path(src))
            if nr is not None:  # NX Studio は調整ファイルの★を優先して表示する
                ph.rating = nr
            ph.focus_result = str(d.get("FocusResult", ""))
            ph.af_mode = str(d.get("AFAreaMode", ""))
            iw, ih = _to_int(d.get("AFImageWidth")), _to_int(d.get("AFImageHeight"))
            ax, ay = _to_int(d.get("AFAreaXPosition")), _to_int(d.get("AFAreaYPosition"))
            aw, ah = _to_int(d.get("AFAreaWidth")), _to_int(d.get("AFAreaHeight"))
            if iw and ih and ax and ay:  # (0,0) は「座標なし」の意味
                aw = aw or iw // 10
                ah = ah or ih // 10
                ph.af_box = (ax / iw, ay / ih, aw / iw, ah / ih)
            for t in PREVIEW_TAGS:
                s, n = _to_int(d.get(t + "Start")), _to_int(d.get(t + "Length"))
                if s and n and n > 1000:
                    ph.previews.append((t, s, n))
            ph.previews.sort(key=lambda x: -x[2])  # 大きい順
            photos[src] = ph
        done += len(chunk)
        if progress:
            progress(done)
    result = []
    for p in paths:
        ph = photos.get(os.path.normpath(p))
        result.append(ph if ph else Photo(path=p, error="メタデータを読めません"))
    return result


# --------------------------------------------------------------------------
# プレビュー読み込み
# --------------------------------------------------------------------------
_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def jpeg_size(b: bytes) -> Optional[Tuple[int, int]]:
    """JPEG のヘッダから (幅, 高さ) を読む（デコード不要）。"""
    if b[:2] != b"\xff\xd8":
        return None
    i, n = 2, len(b)
    while i + 9 < n:
        if b[i] != 0xFF:
            i += 1
            continue
        m = b[i + 1]
        if m == 0xFF:
            i += 1
            continue
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        seg = int.from_bytes(b[i + 2:i + 4], "big")
        if m in _SOF:
            h = int.from_bytes(b[i + 5:i + 7], "big")
            w = int.from_bytes(b[i + 7:i + 9], "big")
            return w, h
        i += 2 + seg
    return None


def load_preview_bytes(photo: Photo, et: Optional[ExifTool] = None) -> Optional[bytes]:
    """内蔵プレビューのうち一番大きい JPEG を返す。

    exiftool で調べた位置からファイルを直接読む（速い）。位置が JPEG の先頭でなければ
    exiftool -b で取り出す。
    """
    for tag, start, length in photo.previews:
        try:
            with open(photo.path, "rb") as f:
                f.seek(start)
                data = f.read(length)
            if data[:2] == b"\xff\xd8":
                photo.preview_tag = tag
                return data
        except OSError:
            pass
        if et is not None:
            out, _ = et.execute(["-b", "-" + tag, photo.path])
            if out[:2] == b"\xff\xd8":
                photo.preview_tag = tag
                return out
    return None


def decode(data: bytes, long_side: Optional[int], gray: bool = True) -> Optional[np.ndarray]:
    """JPEG を目標の長辺サイズでデコードする。縮小が大きいときは JPEG の縮小デコードで高速化。"""
    size = jpeg_size(data)
    flag_full = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    flag = flag_full
    if long_side and size:
        ratio = max(size) / long_side
        if ratio >= 4:
            flag = cv2.IMREAD_REDUCED_GRAYSCALE_4 if gray else cv2.IMREAD_REDUCED_COLOR_4
        elif ratio >= 2:
            flag = cv2.IMREAD_REDUCED_GRAYSCALE_2 if gray else cv2.IMREAD_REDUCED_COLOR_2
    img = cv2.imdecode(np.frombuffer(data, np.uint8), flag)
    if img is None:
        return None
    if long_side:
        h, w = img.shape[:2]
        if max(h, w) > long_side * 1.05:
            s = long_side / max(h, w)
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return img


# --------------------------------------------------------------------------
# ピント判定
# --------------------------------------------------------------------------
def _laplacian(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (3, 3), 0)  # センサーノイズの影響を軽減
    return cv2.Laplacian(g, cv2.CV_32F, ksize=3)


def noise_floor(lap: np.ndarray) -> float:
    """ノイズ由来のラプラシアン分散を推定する。

    細かいブロック（32×24）の分散の下位2%を「平坦部＝ノイズだけの領域」とみなす。
    これを差し引くと、高感度ノイズで「ボケているのにスコアが高い」状態を防げる。
    """
    h, w = lap.shape
    fine = []
    for r in range(24):
        for c in range(32):
            blk = lap[r * h // 24:(r + 1) * h // 24, c * w // 32:(c + 1) * w // 32]
            if blk.size:
                fine.append(float(blk.var()))
    return float(np.percentile(fine, NOISE_FLOOR_PCT)) if fine else 0.0


def global_score(lap: np.ndarray, floor: float = 0.0) -> Tuple[float, Tuple[float, float]]:
    """画像を分割し、シャープなブロック上位の平均を返す（背景ボケに強い）。"""
    h, w = lap.shape
    vals = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            y0, y1 = r * h // GRID_ROWS, (r + 1) * h // GRID_ROWS
            x0, x1 = c * w // GRID_COLS, (c + 1) * w // GRID_COLS
            vals.append((float(lap[y0:y1, x0:x1].var()), ((c + 0.5) / GRID_COLS, (r + 0.5) / GRID_ROWS)))
    vals.sort(key=lambda v: -v[0])
    k = max(3, int(len(vals) * TOP_RATIO))
    return max(0.0, sum(v for v, _ in vals[:k]) / k - floor), vals[0][1]


def af_score(lap: np.ndarray, box: Tuple[float, float, float, float], floor: float = 0.0) -> Optional[float]:
    """AF エリア内（3×3 に分割した上位の平均）のシャープネス。"""
    h, w = lap.shape
    cx, cy, bw, bh = box
    min_side = 0.04 * min(w, h)  # 小さすぎる枠は少し広げる
    pw, ph = max(bw * w, min_side), max(bh * h, min_side)
    x0, x1 = int(max(0, cx * w - pw / 2)), int(min(w, cx * w + pw / 2))
    y0, y1 = int(max(0, cy * h - ph / 2)), int(min(h, cy * h + ph / 2))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    roi = lap[y0:y1, x0:x1]
    rh, rw = roi.shape
    vals = []
    for r in range(3):
        for c in range(3):
            blk = roi[r * rh // 3:(r + 1) * rh // 3, c * rw // 3:(c + 1) * rw // 3]
            if blk.size:
                vals.append(float(blk.var()))
    vals.sort(reverse=True)
    return max(0.0, sum(vals[:4]) / min(4, len(vals)) - floor)


def analyze(photo: Photo, long_side: Optional[int] = None, et: Optional[ExifTool] = None) -> Photo:
    """1枚を解析してスコアを Photo に書き込む。"""
    try:
        if not photo.previews:
            photo.error = photo.error or "内蔵プレビューがありません"
            return photo
        data = load_preview_bytes(photo, et)
        if data is None:
            photo.error = "内蔵プレビューを読めません"
            return photo
        photo.preview_size = jpeg_size(data)
        gray = decode(data, long_side, gray=True)
        if gray is None:
            photo.error = "プレビューをデコードできません"
            return photo
        lap = _laplacian(gray)
        floor = noise_floor(lap)
        photo.score_global, photo.best_block = global_score(lap, floor)
        if photo.af_box:
            photo.score_af = af_score(lap, photo.af_box, floor)
    except Exception as e:  # noqa: BLE001
        photo.error = str(e)
    return photo


def percentile_threshold(scores: Sequence[float], percent: float) -> float:
    return float(np.percentile(np.asarray(scores, dtype=float), percent)) if scores else 0.0


# --------------------------------------------------------------------------
# NX Studio の調整ファイル（NKSC_PARAM/<ファイル名>.nksc）
# --------------------------------------------------------------------------
# NX Studio は調整ファイルがある写真では NEF 本体の★を見ない。表示に使うのは、調整ファイルに
# base64 で埋め込まれた XMP（<ast:XMLPackets>）の xmp:Rating。<nine:Rating> にも★の欄があるので両方そろえる。
# 形式は非公開のため、exiftool で書き直さず（nine 名前空間のタグが消える）★の部分だけを置き換える。
_NKSC_NINE = re.compile(rb"(<nine:Rating>)(\d)(</nine:Rating>)")
_NKSC_PACKET = re.compile(rb"<ast:XMLPackets[^>]*>\s*<rdf:value>([^<]+)</rdf:value>")
_XMP_RATING = (re.compile(rb"(<xmp:Rating>)(-?\d)(</xmp:Rating>)"),
               re.compile(rb'(xmp:Rating=")(-?\d)(")'))
_XMP_NS = b'xmlns:xmp="http://ns.adobe.com/xap/1.0/"'
_B64_NL = b"&#xA;"   # 埋め込み XMP の base64 は &#xA; で改行されている


def nksc_path(path: str) -> Optional[str]:
    """NX Studio の調整ファイルがあればそのパスを返す。"""
    p = os.path.join(os.path.dirname(path), "NKSC_PARAM", os.path.basename(path) + ".nksc")
    return p if os.path.isfile(p) else None


def _nksc_packet(data: bytes):
    """調整ファイルに埋め込まれた XMP を (正規表現のマッチ, XMP のバイト列) で返す。"""
    m = _NKSC_PACKET.search(data)
    if not m:
        return None
    try:
        return m, base64.b64decode(m.group(1).replace(_B64_NL, b""), validate=True)
    except ValueError:
        return None


def _wrap_b64(data: bytes, width: int) -> bytes:
    flat = base64.b64encode(data)
    return _B64_NL.join(flat[i:i + width] for i in range(0, len(flat), width))


def _set_xmp_rating(xmp: bytes, rating: int) -> Optional[bytes]:
    """XMP の★を書き換える。★の項目がなければ、末尾の空白（パディング）を同じ長さだけ削って差し込む。

    XMP の長さを変えないので、base64 の長さや改行位置も変わらない。差し込めなければ None。
    """
    for pat in _XMP_RATING:
        new, k = pat.subn(lambda m: m.group(1) + b"%d" % rating + m.group(3), xmp)
        if k:
            return new
    if rating == 0:  # ★の項目がない＝★0
        return xmp
    tag = b"<xmp:Rating>%d</xmp:Rating>" % rating
    i = xmp.find(_XMP_NS)
    head_end = xmp.find(b">", i)
    j = xmp.find(b"</rdf:Description>", i)
    end = xmp.rfind(b"<?xpacket end")
    if i < 0 or j < 0 or end < j or xmp[head_end - 1:head_end] == b"/" or xmp[end - len(tag):end].strip():
        return None
    return xmp[:j] + tag + xmp[j:end - len(tag)] + xmp[end:]


def read_nksc_rating(nksc: Optional[str]) -> Optional[int]:
    """NX Studio に表示される調整ファイルの★を返す。ファイルや★の欄がなければ None。"""
    if not nksc:
        return None
    try:
        with open(nksc, "rb") as f:
            data = f.read()
    except OSError:
        return None
    pk = _nksc_packet(data)
    if pk:
        for pat in _XMP_RATING:
            m = pat.search(pk[1])
            if m:
                return int(m.group(2))
    m = _NKSC_NINE.search(data)
    if m:
        return int(m.group(2))
    return 0 if pk else None


def write_nksc_rating(nksc: str, rating: int, keep_backup: bool = False) -> str:
    """調整ファイルの★を書き換えて、"updated" / "unchanged" / "notag" のいずれかを返す。

    埋め込み XMP の xmp:Rating と <nine:Rating> の★の部分だけを置き換え、ファイルの長さは変えない。
    書き換えられない形式なら何もせず "notag" を返す。
    同じフォルダの一時ファイルに書いてから置き換える。
    keep_backup=True のとき、元のファイルを <名前>.nksc_original として残す。
    """
    if not 0 <= rating <= 5:
        raise ValueError("rating は 0〜5")
    with open(nksc, "rb") as f:
        data = f.read()
    pk = _nksc_packet(data)
    if not pk:
        return "notag"
    m, xmp = pk
    raw = m.group(1)
    width = len(raw.split(_B64_NL, 1)[0])
    new_xmp = _set_xmp_rating(xmp, rating)
    if new_xmp is None or _wrap_b64(xmp, width) != raw:  # 改行位置を再現できない形式は触らない
        return "notag"
    new = data[:m.start(1)] + _wrap_b64(new_xmp, width) + data[m.end(1):]
    new = _NKSC_NINE.sub(lambda m: m.group(1) + b"%d" % rating + m.group(3), new)
    if new == data:
        return "unchanged"
    if keep_backup:
        shutil.copy2(nksc, nksc + "_original")
    tmp = nksc + ".tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(new)
        shutil.copymode(nksc, tmp)
        os.replace(tmp, nksc)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return "updated"


# --------------------------------------------------------------------------
# レーティング書き込み
# --------------------------------------------------------------------------
@dataclass
class WriteSummary:
    updated: int = 0
    unchanged: int = 0
    skipped: int = 0
    failed: int = 0
    nksc_updated: int = 0     # NX Studio の調整ファイル
    nksc_unchanged: int = 0
    nksc_failed: int = 0
    messages: List[str] = field(default_factory=list)


_SUMMARY_PATTERNS = (
    ("updated", re.compile(r"(\d+) image files? updated")),
    ("unchanged", re.compile(r"(\d+) image files? unchanged")),
    ("skipped", re.compile(r"(\d+) (?:image )?files? failed (?:the )?condition")),
    ("failed", re.compile(r"(\d+) files? weren't updated due to errors")),
)


def write_ratings(et: ExifTool, paths: Sequence[str], rating: int, keep_backup: bool = False,
                  only_unrated: bool = True,
                  progress: Optional[Callable[[int], None]] = None,
                  extra_args: Sequence[str] = (), nksc: bool = True) -> WriteSummary:
    """XMP:Rating を書き込む。画像データは変更しない。

    keep_backup=False のとき -overwrite_original（_original を作らない）。
    only_unrated=True のとき、すでに★が付いているファイルは変更しない。
    nksc=True のとき、NX Studio の調整ファイルがあればその★も書き換える。
    このとき only_unrated は、本体と調整ファイルのどちらかに★があればスキップする。
    -P でファイルの更新日時を保つ。
    """
    if not 0 <= rating <= 5:
        raise ValueError("rating は 0〜5")
    s = WriteSummary()
    todo = list(paths)
    if only_unrated:
        current = read_ratings(et, todo, nksc=False)
        todo = [p for p in todo if not current.get(os.path.normpath(p))
                and not (nksc and read_nksc_rating(nksc_path(p)))]
        s.skipped += len(paths) - len(todo)
    done = len(paths) - len(todo)
    for i in range(0, len(todo), WRITE_CHUNK):
        chunk = todo[i:i + WRITE_CHUNK]
        args = list(extra_args) + ["-P", "-XMP:Rating=%d" % rating]
        if not keep_backup:
            args.insert(0, "-overwrite_original")
        if only_unrated:
            args += ["-if", "not $XMP:Rating"]
        out, err = et.execute(args + chunk)
        text = out.decode("utf-8", "replace")
        for key, pat in _SUMMARY_PATTERNS:
            m = pat.search(text)
            if m:
                setattr(s, key, getattr(s, key) + int(m.group(1)))
        if err:
            s.messages.append(err)
        if nksc:
            for p in chunk:
                _write_nksc(p, rating, keep_backup, s)
        done += len(chunk)
        if progress:
            progress(done)
    return s


def _write_nksc(path: str, rating: int, keep_backup: bool, s: WriteSummary):
    n = nksc_path(path)
    if not n:
        return
    try:
        r = write_nksc_rating(n, rating, keep_backup)
    except OSError as e:
        r = "error: %s" % e
    if r == "updated":
        s.nksc_updated += 1
    elif r == "unchanged":
        s.nksc_unchanged += 1
    else:
        s.nksc_failed += 1
        reason = "★を書き換えられない形式です" if r == "notag" else r
        s.messages.append("NX Studio の調整ファイルを書き換えられません（%s）: %s" % (reason, n))


def read_ratings(et: ExifTool, paths: Sequence[str], nksc: bool = True) -> dict:
    """{パス: ★} を返す（タグなしは 0）。

    nksc=True のとき、NX Studio の調整ファイルに★があればそちらを返す（NX Studio と同じ見え方）。
    """
    res = {}
    for i in range(0, len(paths), META_CHUNK):
        chunk = list(paths[i:i + META_CHUNK])
        out, _ = et.execute(["-json", "-fast", "-XMP:Rating"] + chunk)
        try:
            items = json.loads(out.decode("utf-8", "replace") or "[]")
        except json.JSONDecodeError:
            items = []
        for d in items:
            res[os.path.normpath(d.get("SourceFile", ""))] = _to_int(d.get("Rating")) or 0
    if nksc:
        for p in paths:
            nr = read_nksc_rating(nksc_path(p))
            if nr is not None:
                res[os.path.normpath(p)] = nr
    return res

# -*- coding: utf-8 -*-
"""PySide6 GUI。"""
from __future__ import annotations

import math
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import cv2
import numpy as np
from PySide6.QtCore import (QAbstractTableModel, QModelIndex, QObject, QPointF, QRectF,
                            QSettings, QSortFilterProxyModel, Qt, QThread, Signal)
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QFileDialog,
                               QFrame, QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox,
                               QProgressBar, QPushButton, QSizePolicy, QSpinBox, QSplitter,
                               QTableView, QVBoxLayout, QWidget)

from . import __version__, core

ACCENT = QColor("#E8A33D")
BLUR = QColor("#E5484D")
OK = QColor("#46A758")
APP_NAME = "NEF Focus Rater"


# --------------------------------------------------------------------------
# テーマ
# --------------------------------------------------------------------------
def apply_dark_theme(app: QApplication):
    app.setStyle("Fusion")
    p = QPalette()
    base, alt, win, text = QColor("#1b1d21"), QColor("#22252a"), QColor("#16181b"), QColor("#e6e6e6")
    for role, c in ((QPalette.Window, win), (QPalette.WindowText, text), (QPalette.Base, base),
                    (QPalette.AlternateBase, alt), (QPalette.Text, text), (QPalette.Button, QColor("#2a2d33")),
                    (QPalette.ButtonText, text), (QPalette.ToolTipBase, alt), (QPalette.ToolTipText, text),
                    (QPalette.Highlight, QColor("#3d4a5c")), (QPalette.HighlightedText, QColor("#ffffff")),
                    (QPalette.PlaceholderText, QColor("#7d828a"))):
        p.setColor(role, c)
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#666a70"))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor("#666a70"))
    app.setPalette(p)
    app.setStyleSheet("""
        QPushButton { padding: 6px 14px; border-radius: 6px; border: 1px solid #3a3e45; background: #2a2d33; }
        QPushButton:hover { background: #33373e; }
        QPushButton:disabled { color: #666a70; border-color: #2c2f34; }
        QPushButton#accent { background: #E8A33D; color: #1b1d21; border: none; font-weight: 600; }
        QPushButton#accent:hover { background: #f0b456; }
        QPushButton#accent:disabled { background: #4a4235; color: #8a8070; }
        QTableView { gridline-color: #2a2d33; border: 1px solid #2a2d33; border-radius: 6px; }
        QHeaderView::section { background: #22252a; padding: 4px 6px; border: none;
                               border-right: 1px solid #2a2d33; color: #b8bcc4; }
        QComboBox, QSpinBox { padding: 3px 6px; border-radius: 5px; border: 1px solid #3a3e45; background: #22252a; }
        QProgressBar { border: none; background: #22252a; border-radius: 3px; height: 6px; }
        QProgressBar::chunk { background: #E8A33D; border-radius: 3px; }
        QLabel#muted { color: #9aa0a8; }
        QFrame#panel { background: #1b1d21; border: 1px solid #2a2d33; border-radius: 8px; }
    """)


# --------------------------------------------------------------------------
# 画像ヘルパ
# --------------------------------------------------------------------------
def to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def rotate_for_display(img: np.ndarray, orientation: int) -> np.ndarray:
    if orientation == 6:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 8:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if orientation == 3:
        return cv2.rotate(img, cv2.ROTATE_180)
    return img


def draw_box(img: np.ndarray, box, color=(61, 163, 232), thickness=None):
    h, w = img.shape[:2]
    cx, cy, bw, bh = box
    t = thickness or max(2, int(min(w, h) / 300))
    x0, y0 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
    x1, y1 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, t)


# --------------------------------------------------------------------------
# テーブルモデル
# --------------------------------------------------------------------------
COLS = ["書込", "ファイル", "判定スコア", "AFスコア", "全体スコア", "現在の★", "カメラAF", "プレビュー", "判定"]
C_CHECK, C_NAME, C_SCORE, C_AF, C_GLOBAL, C_RATING, C_CAMAF, C_PREV, C_VERDICT = range(len(COLS))


class PhotoModel(QAbstractTableModel):
    targetsChanged = Signal()

    def __init__(self):
        super().__init__()
        self.photos: List[core.Photo] = []
        self.base = ""
        self.threshold = 0.0
        self.prefer_af = True
        self.overrides = {}  # path -> bool（手動で書込対象を切り替えたもの）

    def set_photos(self, photos, base):
        self.beginResetModel()
        self.photos, self.base, self.overrides = photos, base, {}
        self.endResetModel()

    def score(self, p: core.Photo):
        return p.score(self.prefer_af)

    def is_blur(self, p: core.Photo) -> bool:
        s = self.score(p)
        return s is not None and s < self.threshold

    def is_target(self, p: core.Photo) -> bool:
        if p.path in self.overrides:
            return self.overrides[p.path]
        return self.is_blur(p)

    def targets(self) -> List[core.Photo]:
        return [p for p in self.photos if self.score(p) is not None and self.is_target(p)]

    def refresh(self):
        if self.photos:
            self.dataChanged.emit(self.index(0, 0), self.index(len(self.photos) - 1, len(COLS) - 1))
        self.targetsChanged.emit()

    # Qt API
    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.photos)

    def columnCount(self, parent=QModelIndex()):
        return len(COLS)

    def headerData(self, section, orient, role=Qt.DisplayRole):
        if orient == Qt.Horizontal and role == Qt.DisplayRole:
            return COLS[section]
        return None

    def flags(self, index):
        f = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if index.column() == C_CHECK and self.score(self.photos[index.row()]) is not None:
            f |= Qt.ItemIsUserCheckable
        return f

    def data(self, index, role=Qt.DisplayRole):
        p = self.photos[index.row()]
        c = index.column()
        s = self.score(p)
        if role == Qt.CheckStateRole and c == C_CHECK:
            if s is None:
                return None
            return Qt.Checked if self.is_target(p) else Qt.Unchecked
        if role in (Qt.DisplayRole, Qt.UserRole):
            sort = role == Qt.UserRole
            if c == C_NAME:
                return os.path.relpath(p.path, self.base) if self.base else p.name
            if c == C_SCORE:
                return (s if s is not None else -1) if sort else ("—" if s is None else "%.1f" % s)
            if c == C_AF:
                v = p.score_af
                return (v if v is not None else -1) if sort else ("—" if v is None else "%.1f" % v)
            if c == C_GLOBAL:
                v = p.score_global
                return (v if v is not None else -1) if sort else ("—" if v is None else "%.1f" % v)
            if c == C_RATING:
                return (p.rating or 0) if sort else ("—" if p.rating is None else ("★" * p.rating if p.rating else "0"))
            if c == C_CAMAF:
                return {"Focus": "合焦", "Out of Focus": "非合焦"}.get(p.focus_result, p.focus_result or "—")
            if c == C_PREV:
                if not p.preview_size:
                    return 0 if sort else "—"
                w, h = p.preview_size
                return w * h if sort else "%d×%d" % (w, h)
            if c == C_VERDICT:
                if p.error:
                    return "エラー" if not sort else 0
                return ("ピンボケ" if self.is_blur(p) else "OK") if not sort else (1 if self.is_blur(p) else 2)
            if c == C_CHECK:
                return (1 if self.is_target(p) else 0) if sort else None
        if role == Qt.ForegroundRole:
            if c == C_VERDICT:
                if p.error:
                    return QColor("#9aa0a8")
                return BLUR if self.is_blur(p) else OK
            if c == C_PREV and p.preview_size and max(p.preview_size) < 3000:
                return ACCENT  # プレビューが小さい（判定精度が落ちる）
        if role == Qt.BackgroundRole and s is not None and self.is_target(p):
            return QColor(229, 72, 77, 40)
        if role == Qt.TextAlignmentRole and c not in (C_NAME,):
            return Qt.AlignCenter
        if role == Qt.ToolTipRole:
            if p.error:
                return p.error
            if c == C_PREV:
                return "使用したプレビュー: %s" % (p.preview_tag or "—")
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if role == Qt.CheckStateRole and index.column() == C_CHECK:
            p = self.photos[index.row()]
            checked = (value == Qt.Checked or value == Qt.Checked.value)
            if checked == self.is_blur(p):
                self.overrides.pop(p.path, None)
            else:
                self.overrides[p.path] = checked
            r = index.row()
            self.dataChanged.emit(self.index(r, 0), self.index(r, len(COLS) - 1))
            self.targetsChanged.emit()
            return True
        return False


# --------------------------------------------------------------------------
# ヒストグラム（クリック／ドラッグでしきい値を設定）
# --------------------------------------------------------------------------
class Histogram(QWidget):
    thresholdChanged = Signal(float)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(90)
        self.logs: List[float] = []
        self.lo, self.hi = 0.0, 1.0
        self.th = 0.0
        self.setCursor(Qt.CrossCursor)
        self.setToolTip("クリックまたはドラッグでしきい値を設定")

    @staticmethod
    def to_log(v: float) -> float:
        return math.log10(1.0 + max(0.0, v))

    def set_scores(self, scores: List[float]):
        self.logs = [self.to_log(s) for s in scores]
        if self.logs:
            self.lo, self.hi = min(self.logs), max(self.logs)
            if self.hi - self.lo < 1e-6:
                self.hi = self.lo + 1.0
            pad = (self.hi - self.lo) * 0.03
            self.lo, self.hi = self.lo - pad, self.hi + pad
        self.update()

    def set_threshold(self, value: float):
        self.th = self.to_log(value)
        self.update()

    def _x_to_log(self, x: float) -> float:
        r = self.rect().adjusted(8, 8, -8, -20)
        t = min(1.0, max(0.0, (x - r.left()) / max(1, r.width())))
        return self.lo + t * (self.hi - self.lo)

    def mousePressEvent(self, e):
        self._emit(e.position().x())

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            self._emit(e.position().x())

    def _emit(self, x):
        if not self.logs:
            return
        self.th = self._x_to_log(x)
        self.update()
        self.thresholdChanged.emit(10 ** self.th - 1.0)

    def paintEvent(self, _):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect().adjusted(8, 8, -8, -20))
        qp.fillRect(self.rect(), QColor("#1b1d21"))
        if not self.logs:
            qp.setPen(QColor("#7d828a"))
            qp.drawText(self.rect(), Qt.AlignCenter, "解析するとスコアの分布が表示されます")
            return
        nb = max(10, min(60, int(r.width() / 8)))
        counts = [0] * nb
        for v in self.logs:
            i = int((v - self.lo) / (self.hi - self.lo) * nb)
            counts[min(nb - 1, max(0, i))] += 1
        mx = max(counts) or 1
        bw = r.width() / nb
        for i, n in enumerate(counts):
            if not n:
                continue
            x = r.left() + i * bw
            h = (n / mx) * r.height()
            center = self.lo + (i + 0.5) / nb * (self.hi - self.lo)
            qp.fillRect(QRectF(x + 1, r.bottom() - h, bw - 2, h), BLUR if center < self.th else QColor("#5b6472"))
        tx = r.left() + (self.th - self.lo) / (self.hi - self.lo) * r.width()
        qp.setPen(QPen(ACCENT, 2))
        qp.drawLine(QPointF(tx, r.top() - 4), QPointF(tx, r.bottom() + 2))
        qp.setPen(QColor("#9aa0a8"))
        f = qp.font()
        f.setPointSizeF(max(8.0, f.pointSizeF() - 1))
        qp.setFont(f)
        qp.drawText(QRectF(r.left(), r.bottom() + 2, 200, 16), Qt.AlignLeft, "← ボケている")
        qp.drawText(QRectF(r.right() - 200, r.bottom() + 2, 200, 16), Qt.AlignRight, "シャープ →")


# --------------------------------------------------------------------------
# プレビュー表示
# --------------------------------------------------------------------------
class ImageView(QLabel):
    def __init__(self, placeholder: str):
        super().__init__(placeholder)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pix: Optional[QPixmap] = None
        self.setObjectName("muted")

    def set_image(self, img: Optional[QImage], text: str = ""):
        self._pix = QPixmap.fromImage(img) if img is not None else None
        if self._pix is None:
            self.setPixmap(QPixmap())
            self.setText(text)
        self._rescale()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale()

    def _rescale(self):
        if self._pix is not None:
            self.setPixmap(self._pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class PreviewLoader(QObject):
    """選択された写真のプレビューと等倍切り出しをバックグラウンドで作る（最新の要求だけ処理）。"""
    loaded = Signal(int, QImage, QImage)
    failed = Signal(int, str)

    def __init__(self):
        super().__init__()
        self._ex = ThreadPoolExecutor(max_workers=1)
        self._req = 0
        self.et: Optional[core.ExifTool] = None

    def request(self, photo: core.Photo):
        self._req += 1
        rid = self._req
        self._ex.submit(self._load, rid, photo)

    def _load(self, rid, p: core.Photo):
        if rid != self._req:
            return
        try:
            data = core.load_preview_bytes(p, self.et)
            if data is None:
                self.failed.emit(rid, "プレビューを読めません")
                return
            full = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if full is None:
                self.failed.emit(rid, "デコードできません")
                return
            h, w = full.shape[:2]
            # 全体表示（縮小）
            s = 1600 / max(h, w)
            small = cv2.resize(full, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else full.copy()
            if p.af_box:
                draw_box(small, p.af_box)
            # 等倍切り出し（AFエリア、なければ最もシャープなブロック）
            if p.af_box:
                cx, cy = p.af_box[0] * w, p.af_box[1] * h
            elif p.best_block:
                cx, cy = p.best_block[0] * w, p.best_block[1] * h
            else:
                cx, cy = w / 2, h / 2
            cw, ch = min(w, 900), min(h, 600)
            x0 = int(min(max(0, cx - cw / 2), w - cw))
            y0 = int(min(max(0, cy - ch / 2), h - ch))
            crop = full[y0:y0 + ch, x0:x0 + cw].copy()
            if p.af_box:
                bx, by, bw, bh = p.af_box
                rel = ((bx * w - x0) / cw, (by * h - y0) / ch, bw * w / cw, bh * h / ch)
                draw_box(crop, rel, thickness=2)
            small = rotate_for_display(small, p.orientation)
            crop = rotate_for_display(crop, p.orientation)
            if rid == self._req:
                self.loaded.emit(rid, to_qimage(small), to_qimage(crop))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(rid, str(e))


# --------------------------------------------------------------------------
# バックグラウンド処理
# --------------------------------------------------------------------------
class AnalyzeWorker(QThread):
    progress = Signal(int, int, str)
    finished_ok = Signal(list)
    failed = Signal(str)

    def __init__(self, et: core.ExifTool, files: List[str], long_side: Optional[int]):
        super().__init__()
        self.et, self.files, self.long_side = et, files, long_side
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        try:
            n = len(self.files)
            self.progress.emit(0, n, "メタデータを読み込み中…")
            photos = core.read_metadata(self.et, self.files,
                                        progress=lambda d: self.progress.emit(d, n, "メタデータを読み込み中…"))
            done = 0
            workers = min(8, os.cpu_count() or 4)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(self._one, p) for p in photos]
                for _f in futs:
                    _f.result()
                    done += 1
                    self.progress.emit(done, n, "ピントを解析中…")
            if self._cancel.is_set():
                photos = [p for p in photos if p.score_global is not None or p.error]
            self.finished_ok.emit(photos)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))

    def _one(self, p):
        if not self._cancel.is_set():
            core.analyze(p, self.long_side, self.et)


class WriteWorker(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(object, dict)
    failed = Signal(str)

    def __init__(self, et, paths, rating, keep_backup, only_unrated, nksc):
        super().__init__()
        self.et, self.paths = et, paths
        self.rating, self.keep_backup, self.only_unrated = rating, keep_backup, only_unrated
        self.nksc = nksc

    def run(self):
        try:
            n = len(self.paths)
            s = core.write_ratings(self.et, self.paths, self.rating, self.keep_backup, self.only_unrated,
                                   progress=lambda d: self.progress.emit(d, n), nksc=self.nksc)
            back = core.read_ratings(self.et, self.paths, self.nksc)
            self.finished_ok.emit(s, back)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


# --------------------------------------------------------------------------
# メインウィンドウ
# --------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("%s %s" % (APP_NAME, __version__))
        self.resize(1360, 860)
        self.settings = QSettings("nef-focus-rater", "nef-focus-rater")
        self.et: Optional[core.ExifTool] = None
        self.worker = None
        self.busy = False
        self.folder = ""

        self.model = PhotoModel()
        self.proxy = QSortFilterProxyModel()
        self.proxy.setSourceModel(self.model)
        self.proxy.setSortRole(Qt.UserRole)
        self.loader = PreviewLoader()
        self.loader.loaded.connect(self._on_preview)
        self.loader.failed.connect(self._on_preview_failed)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 12, 14, 8)
        root.setSpacing(10)

        # --- 上段：フォルダと解析設定
        top = QHBoxLayout()
        self.btn_folder = QPushButton("フォルダを選択…")
        self.btn_folder.clicked.connect(self.choose_folder)
        self.lbl_folder = QLabel("NEF の入ったフォルダを選んでください")
        self.lbl_folder.setObjectName("muted")
        self.lbl_folder.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.chk_recursive = QCheckBox("サブフォルダも含める")
        self.cmb_res = QComboBox()
        self.cmb_res.addItem("解析: フル解像度", "full")
        self.cmb_res.addItem("解析: 1/2（長辺3000px）", "half")
        self.cmb_res.addItem("解析: 1/4（長辺1500px・高速）", "quarter")
        self.cmb_res.setToolTip("フル解像度ほど細かなピントのズレを検出できます。\n"
                                "高感度ノイズが多い写真は 1/2 のほうが安定する場合があります。")
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItem("判定: AFエリア優先", True)
        self.cmb_mode.addItem("判定: 画像全体", False)
        self.cmb_mode.setToolTip("AFエリア優先: カメラが記録したAFエリア内のシャープさで判定します\n"
                                 "（AF情報がない写真は画像全体で判定）")
        self.cmb_mode.currentIndexChanged.connect(self._on_mode)
        self.btn_analyze = QPushButton("解析開始")
        self.btn_analyze.setObjectName("accent")
        self.btn_analyze.clicked.connect(self.toggle_analyze)
        top.addWidget(self.btn_folder)
        top.addWidget(self.lbl_folder, 1)
        top.addWidget(self.chk_recursive)
        top.addWidget(self.cmb_res)
        top.addWidget(self.cmb_mode)
        top.addWidget(self.btn_analyze)
        root.addLayout(top)

        # --- 中段：一覧とプレビュー
        split = QSplitter(Qt.Horizontal)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(C_SCORE, Qt.AscendingOrder)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(26)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(C_NAME, QHeaderView.Stretch)
        self.table.selectionModel().currentRowChanged.connect(self._on_row)
        split.addWidget(self.table)

        right = QFrame()
        right.setObjectName("panel")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(10, 10, 10, 10)
        self.view_full = ImageView("一覧から写真を選ぶとプレビューを表示します")
        self.view_crop = ImageView("")
        lbl_crop = QLabel("等倍（AFエリア／最もシャープな部分）")
        lbl_crop.setObjectName("muted")
        self.lbl_info = QLabel("")
        self.lbl_info.setObjectName("muted")
        self.lbl_info.setWordWrap(True)
        rl.addWidget(self.view_full, 3)
        rl.addWidget(lbl_crop)
        rl.addWidget(self.view_crop, 2)
        rl.addWidget(self.lbl_info)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        root.addWidget(split, 1)

        # --- しきい値
        thr = QFrame()
        thr.setObjectName("panel")
        tl = QVBoxLayout(thr)
        tl.setContentsMargins(10, 8, 10, 8)
        head = QHBoxLayout()
        t = QLabel("ピンボケ判定のしきい値")
        f = t.font()
        f.setBold(True)
        t.setFont(f)
        head.addWidget(t)
        self.lbl_thr = QLabel("")
        self.lbl_thr.setObjectName("muted")
        head.addWidget(self.lbl_thr, 1)
        head.addWidget(QLabel("下位"))
        self.spin_pct = QSpinBox()
        self.spin_pct.setRange(0, 100)
        self.spin_pct.setSuffix(" %")
        self.spin_pct.setValue(15)
        self.spin_pct.setToolTip("スコアの低い順に、この割合をピンボケ判定にします")
        self.spin_pct.valueChanged.connect(self._on_pct)
        head.addWidget(self.spin_pct)
        tl.addLayout(head)
        self.hist = Histogram()
        self.hist.thresholdChanged.connect(self._on_hist)
        tl.addWidget(self.hist)
        root.addWidget(thr)

        # --- 下段：書き込み
        bot = QHBoxLayout()
        bot.addWidget(QLabel("書き込む★"))
        self.spin_rating = QSpinBox()
        self.spin_rating.setRange(1, 5)
        bot.addWidget(self.spin_rating)
        bot.addSpacing(12)
        self.chk_unrated = QCheckBox("すでに★がある写真は変更しない")
        self.chk_jpeg = QCheckBox("同名の JPEG にも書き込む（RAW+JPEG）")
        self.chk_backup = QCheckBox("元ファイル（_original）を残す")
        self.chk_nksc = QCheckBox("NX Studio の調整ファイルにも書き込む")
        self.chk_nksc.setToolTip("NKSC_PARAM フォルダの調整ファイル（.nksc）がある写真は、NX Studio がそちらの★を表示します。\n"
                                 "★の数字だけを書き換え、編集内容には触れません。")
        for w in (self.chk_unrated, self.chk_jpeg, self.chk_nksc, self.chk_backup):
            bot.addWidget(w)
        bot.addStretch(1)
        self.lbl_targets = QLabel("")
        self.lbl_targets.setObjectName("muted")
        bot.addWidget(self.lbl_targets)
        self.btn_write = QPushButton("★を書き込む")
        self.btn_write.setObjectName("accent")
        self.btn_write.setEnabled(False)
        self.btn_write.clicked.connect(self.do_write)
        bot.addWidget(self.btn_write)
        root.addLayout(bot)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.statusBar().addPermanentWidget(self.progress, 1)
        self.statusBar().showMessage("フォルダを選んで「解析開始」を押してください")
        self.progress.hide()

        self.model.targetsChanged.connect(self._update_targets)
        act = QAction("書込対象を切り替え", self)
        act.setShortcut(QKeySequence(Qt.Key_Space))
        act.setShortcutContext(Qt.WidgetShortcut)
        act.triggered.connect(self._toggle_current)
        self.table.addAction(act)

        self._load_settings()

    # -- 設定
    def _load_settings(self):
        s = self.settings
        self.folder = s.value("folder", "", str)
        if self.folder:
            self.lbl_folder.setText(self.folder)
        self.chk_recursive.setChecked(s.value("recursive", True, bool))
        self.cmb_res.setCurrentIndex(max(0, self.cmb_res.findData(s.value("resolution", "full", str))))
        self.cmb_mode.setCurrentIndex(0 if s.value("prefer_af", True, bool) else 1)
        self.spin_pct.setValue(s.value("percent", 15, int))
        self.spin_rating.setValue(s.value("rating", 1, int))
        self.chk_unrated.setChecked(s.value("only_unrated", True, bool))
        self.chk_jpeg.setChecked(s.value("with_jpeg", True, bool))
        self.chk_backup.setChecked(s.value("keep_backup", False, bool))
        self.chk_nksc.setChecked(s.value("with_nksc", True, bool))
        geo = s.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)

    def _save_settings(self):
        s = self.settings
        s.setValue("folder", self.folder)
        s.setValue("recursive", self.chk_recursive.isChecked())
        s.setValue("resolution", self.cmb_res.currentData())
        s.setValue("prefer_af", bool(self.cmb_mode.currentData()))
        s.setValue("percent", self.spin_pct.value())
        s.setValue("rating", self.spin_rating.value())
        s.setValue("only_unrated", self.chk_unrated.isChecked())
        s.setValue("with_jpeg", self.chk_jpeg.isChecked())
        s.setValue("keep_backup", self.chk_backup.isChecked())
        s.setValue("with_nksc", self.chk_nksc.isChecked())
        s.setValue("geometry", self.saveGeometry())

    def closeEvent(self, e):
        if self.worker and self.worker.isRunning():
            if isinstance(self.worker, WriteWorker):
                QMessageBox.information(self, APP_NAME, "書き込み中です。完了するまでお待ちください。")
                e.ignore()
                return
            self.worker.cancel()
            self.worker.wait(5000)
        self._save_settings()
        if self.et:
            self.et.close()
        super().closeEvent(e)

    # -- exiftool
    def _ensure_et(self) -> bool:
        if self.et:
            return True
        try:
            self.et = core.ExifTool()
            self.loader.et = self.et
            return True
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "exiftool が見つかりません",
                                 "%s\n\nmacOS: brew install exiftool\nWindows: exiftool.org から入手し、"
                                 "exiftool.exe を PATH の通った場所に置いてください。" % e)
            return False

    # -- 操作
    def choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "NEF の入ったフォルダを選択", self.folder or os.path.expanduser("~"))
        if d:
            self.folder = d
            self.lbl_folder.setText(d)

    def toggle_analyze(self):
        if isinstance(self.worker, AnalyzeWorker) and self.worker.isRunning():
            self.worker.cancel()
            self.btn_analyze.setEnabled(False)
            self.statusBar().showMessage("中止しています…")
            return
        if not self.folder or not os.path.isdir(self.folder):
            QMessageBox.warning(self, APP_NAME, "フォルダを選択してください。")
            return
        if not self._ensure_et():
            return
        files = core.list_raw_files(self.folder, self.chk_recursive.isChecked())
        if not files:
            QMessageBox.information(self, APP_NAME, "このフォルダに NEF ファイルがありません。")
            return
        self.model.set_photos([], self.folder)
        self.hist.set_scores([])
        self.view_full.set_image(None, "解析中…")
        self.view_crop.set_image(None, "")
        self.btn_write.setEnabled(False)
        self.btn_analyze.setText("中止")
        self._busy(True, len(files))
        self.worker = AnalyzeWorker(self.et, files, core.RESOLUTIONS[self.cmb_res.currentData()])
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._analysis_done)
        self.worker.failed.connect(self._worker_failed)
        self.worker.start()

    def _busy(self, on: bool, total: int = 0):
        self.busy = on
        self.progress.setVisible(on)
        if on:
            self.progress.setRange(0, max(1, total))
            self.progress.setValue(0)
        for w in (self.btn_folder, self.chk_recursive, self.cmb_res):
            w.setEnabled(not on)

    def _on_progress(self, done, total, msg):
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(done)
        self.statusBar().showMessage("%s %d / %d" % (msg, done, total))

    def _worker_failed(self, msg):
        self._busy(False)
        self._update_targets()
        self.btn_analyze.setText("解析開始")
        self.btn_analyze.setEnabled(True)
        QMessageBox.critical(self, APP_NAME, msg)

    def _analysis_done(self, photos):
        self._busy(False)
        self.btn_analyze.setText("解析開始")
        self.btn_analyze.setEnabled(True)
        self.model.prefer_af = bool(self.cmb_mode.currentData())
        self.model.set_photos(photos, self.folder)
        self._apply_percent()
        n_err = sum(1 for p in photos if p.error)
        n_af = sum(1 for p in photos if p.score_af is not None)
        sizes = sorted({p.preview_size for p in photos if p.preview_size}, key=lambda s: -s[0] * s[1])
        size_txt = "、".join("%d×%d" % s for s in sizes[:3]) or "—"
        self.statusBar().showMessage("解析完了: %d 枚（AFエリア情報あり %d 枚・エラー %d 枚）／ プレビュー解像度: %s"
                                     % (len(photos), n_af, n_err, size_txt))
        self.table.sortByColumn(C_SCORE, Qt.AscendingOrder)
        if self.proxy.rowCount():
            self.table.selectRow(0)

    def _scores(self):
        return [s for s in (self.model.score(p) for p in self.model.photos) if s is not None]

    def _apply_percent(self):
        scores = self._scores()
        self.hist.set_scores(scores)
        if scores:
            pct = self.spin_pct.value()
            if pct <= 0:
                th = -1.0
            else:
                # 下位 p% を確実に含めるため、ほんの少しだけ上げる
                th = core.percentile_threshold(scores, pct) * 1.000001 + 1e-9
            self._set_threshold(th, from_pct=True)

    def _on_pct(self, _):
        self._apply_percent()

    def _on_hist(self, value):
        self._set_threshold(value)

    def _on_mode(self, _):
        self.model.prefer_af = bool(self.cmb_mode.currentData())
        if self.model.photos:
            self._apply_percent()

    def _set_threshold(self, value, from_pct=False):
        self.model.threshold = value
        self.hist.set_threshold(value)
        scores = self._scores()
        n_blur = sum(1 for s in scores if s < value)
        if not from_pct and scores:
            self.spin_pct.blockSignals(True)
            self.spin_pct.setValue(round(100 * n_blur / len(scores)))
            self.spin_pct.blockSignals(False)
        self.lbl_thr.setText("  スコア %.1f 未満 → ピンボケ判定 %d 枚 / %d 枚" % (value, n_blur, len(scores)))
        self.model.refresh()

    def _update_targets(self):
        n = len(self.model.targets())
        self.lbl_targets.setText("書込対象 %d 枚" % n)
        self.btn_write.setEnabled(n > 0 and not self.busy)

    def _toggle_current(self):
        idx = self.table.currentIndex()
        if not idx.isValid():
            return
        src = self.proxy.mapToSource(idx)
        p = self.model.photos[src.row()]
        if self.model.score(p) is None:
            return
        ci = self.model.index(src.row(), C_CHECK)
        self.model.setData(ci, Qt.Unchecked if self.model.is_target(p) else Qt.Checked, Qt.CheckStateRole)

    def _current_photo(self) -> Optional[core.Photo]:
        idx = self.table.currentIndex()
        if not idx.isValid():
            return None
        return self.model.photos[self.proxy.mapToSource(idx).row()]

    def _on_row(self, cur, _prev):
        if not cur.isValid():
            return
        p = self.model.photos[self.proxy.mapToSource(cur).row()]
        self._show_info(p)
        if p.previews:
            self.view_full.set_image(None, "読み込み中…")
            self.loader.request(p)
        else:
            self.view_full.set_image(None, p.error or "プレビューがありません")
            self.view_crop.set_image(None, "")

    def _show_info(self, p: core.Photo):
        def f(v):
            return "—" if v is None else "%.1f" % v
        lines = ["<b>%s</b>" % p.name,
                 "判定スコア %s ／ AF %s ／ 全体 %s" % (f(self.model.score(p)), f(p.score_af), f(p.score_global)),
                 "カメラ: %s ／ AFエリアモード: %s ／ カメラAF結果: %s" % (p.model or "—", p.af_mode or "—",
                                                               p.focus_result or "—"),
                 "プレビュー: %s %s" % (p.preview_tag or "—",
                                     "%d×%d" % p.preview_size if p.preview_size else "")]
        if p.error:
            lines.append("<span style='color:#E5484D'>%s</span>" % p.error)
        self.lbl_info.setText("<br>".join(lines))

    def _on_preview(self, _rid, full: QImage, crop: QImage):
        self.view_full.set_image(full)
        self.view_crop.set_image(crop)

    def _on_preview_failed(self, _rid, msg):
        self.view_full.set_image(None, msg)
        self.view_crop.set_image(None, "")

    # -- 書き込み
    def do_write(self):
        targets = self.model.targets()
        if not targets:
            return
        paths = [p.path for p in targets]
        n_jpeg = 0
        if self.chk_jpeg.isChecked():
            for p in targets:
                j = core.sibling_jpeg(p.path)
                if j:
                    paths.append(j)
                    n_jpeg += 1
        rating = self.spin_rating.value()
        with_nksc = self.chk_nksc.isChecked()
        n_nksc = sum(1 for p in paths if core.nksc_path(p)) if with_nksc else 0
        msg = ("%d 枚の NEF%s に ★%d を書き込みます。\n\n"
               "・画像データは変更せず、レーティング（メタデータ）だけを書き換えます\n"
               "・元ファイル（_original）: %s\n"
               "・すでに★がある写真: %s\n%s\n実行しますか？") % (
            len(targets), "（＋同名 JPEG %d 枚）" % n_jpeg if n_jpeg else "", rating,
            "残す" if self.chk_backup.isChecked() else "残さない",
            "変更しない" if self.chk_unrated.isChecked() else "上書きする",
            ("・NX Studio の調整ファイル %d 個の★も書き換えます。NX Studio を終了してから実行してください\n" % n_nksc)
            if n_nksc else "")
        if QMessageBox.question(self, "書き込みの確認", msg) != QMessageBox.Yes:
            return
        self.btn_write.setEnabled(False)
        self.btn_analyze.setEnabled(False)
        self._busy(True, len(paths))
        self.statusBar().showMessage("書き込み中…")
        self.worker = WriteWorker(self.et, paths, rating, self.chk_backup.isChecked(), self.chk_unrated.isChecked(),
                                  with_nksc)
        self.worker.progress.connect(lambda d, n: self._on_progress(d, n, "書き込み中…"))
        self.worker.finished_ok.connect(self._write_done)
        self.worker.failed.connect(self._worker_failed)
        self.worker.start()

    def _write_done(self, s: core.WriteSummary, back: dict):
        self._busy(False)
        self.btn_analyze.setEnabled(True)
        for p in self.model.photos:
            r = back.get(os.path.normpath(p.path))
            if r is not None:
                p.rating = r
        self.model.refresh()
        text = "更新 %d 枚 ／ ★済みのためスキップ %d 枚 ／ 変更なし %d 枚 ／ エラー %d 枚" % (
            s.updated, s.skipped, s.unchanged, s.failed)
        if s.nksc_updated or s.nksc_unchanged or s.nksc_failed:
            text += "\nNX Studio の調整ファイル: 更新 %d ／ 変更なし %d ／ エラー %d" % (
                s.nksc_updated, s.nksc_unchanged, s.nksc_failed)
        self.statusBar().showMessage("書き込み完了: " + text.replace("\n", " ／ "))
        detail = text
        if s.messages:
            detail += "\n\nexiftool のメッセージ:\n" + "\n".join(s.messages)[:2000]
        (QMessageBox.warning if s.failed or s.nksc_failed else QMessageBox.information)(self, "書き込み完了", detail)
        self._update_targets()


def main():
    QApplication.setApplicationName(APP_NAME)
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

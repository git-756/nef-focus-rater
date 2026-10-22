# -*- coding: utf-8 -*-
"""コマンドライン版（解析だけ。書き込みはしない）。"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from . import __version__, core


def scan(folder: str, recursive: bool, resolution: str, percent: float, csv_path: str = None) -> int:
    files = core.list_raw_files(folder, recursive)
    if not files:
        print("NEF ファイルがありません: %s" % folder, file=sys.stderr)
        return 1
    print("%d 枚を解析します…" % len(files), file=sys.stderr)
    with core.ExifTool() as et:
        photos = core.read_metadata(et, files)
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
            list(ex.map(lambda p: core.analyze(p, core.RESOLUTIONS[resolution], et), photos))
    ok = sorted([p for p in photos if p.score() is not None], key=lambda p: p.score())
    th = core.percentile_threshold([p.score() for p in ok], percent) if ok else 0
    rows = []
    for p in ok + [p for p in photos if p.score() is None]:
        rows.append({
            "file": os.path.relpath(p.path, folder),
            "score": "" if p.score() is None else "%.2f" % p.score(),
            "af_score": "" if p.score_af is None else "%.2f" % p.score_af,
            "global_score": "" if p.score_global is None else "%.2f" % p.score_global,
            "rating": "" if p.rating is None else p.rating,
            "camera_af": p.focus_result,
            "preview": "%s %dx%d" % ((p.preview_tag,) + p.preview_size) if p.preview_size else "",
            "verdict": p.error or ("blur" if p.score() is not None and p.score() <= th else "ok"),
        })
    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("CSV を保存しました: %s" % csv_path, file=sys.stderr)
    else:
        for r in rows:
            print("%10s  %-6s ★%-2s %-12s %s" % (r["score"], r["verdict"], r["rating"], r["preview"], r["file"]))
    print("下位 %g%% のしきい値: %.2f" % (percent, th), file=sys.stderr)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nef-focus-rater",
                                 description="NEF のピントを解析して、ピンボケ写真にレーティングを付けるツール")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd")
    sc = sub.add_parser("scan", help="解析してスコアを表示する（書き込みはしない）")
    sc.add_argument("folder")
    sc.add_argument("--no-recursive", action="store_true", help="サブフォルダを含めない")
    sc.add_argument("--resolution", choices=list(core.RESOLUTIONS), default="full")
    sc.add_argument("--percent", type=float, default=15, help="下位◯%%をピンボケ判定にする（既定 15）")
    sc.add_argument("--csv", help="結果を CSV に保存する")
    a = ap.parse_args(argv)
    if a.cmd == "scan":
        return scan(a.folder, not a.no_recursive, a.resolution, a.percent, a.csv)
    from .gui import main as gui_main  # PySide6 は GUI を使うときだけ読み込む
    return gui_main()


if __name__ == "__main__":
    sys.exit(main())

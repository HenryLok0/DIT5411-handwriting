"""Build gzip TFRecords: first 40 photos train, the rest test, train set expanded to 200."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import write_tfrecords


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main() -> int:
    configure_stdio()
    if sys.version_info >= (3, 14):
        print("TensorFlow 目前沒有 Python 3.14 的安裝包。請改用 Python 3.9 到 3.13。")
        return 1

    parser = argparse.ArgumentParser(description="Split, augment, and write TFRecords.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--zip", type=Path, help="合併後的 all_data.zip")
    source.add_argument("--data-dir", type=Path, help="已經依字分好資料夾的圖片")
    parser.add_argument("--out", type=Path, default=Path("data/records"))
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--work-size", type=int, default=128)
    parser.add_argument("--samples-per-class", type=int, default=200)
    parser.add_argument("--max-classes", type=int, default=0, help="只取前 N 個字。正式作業請保持 0。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index-only", action="store_true", help="只統計類別與張數，不寫圖片。")
    args = parser.parse_args()

    if args.samples_per_class < 200 and args.max_classes == 0:
        print("警告：作業要求每個字至少 200 個訓練樣本。")
    work_size = max(args.work_size, args.img_size)
    write_tfrecords(
        output_dir=args.out,
        img_size=args.img_size,
        work_size=work_size,
        target_per_class=args.samples_per_class,
        seed=args.seed,
        max_classes=args.max_classes,
        zip_path=args.zip,
        data_dir=args.data_dir,
        index_only=args.index_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

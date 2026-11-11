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
        print("TensorFlow has no install package for Python 3.14. Use Python 3.9 through 3.13.")
        return 1

    parser = argparse.ArgumentParser(description="Split, augment, and write TFRecords.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--zip", type=Path, help="Merged all_data.zip")
    source.add_argument("--data-dir", type=Path, help="Images already sorted into one folder per character")
    parser.add_argument("--out", type=Path, default=Path("data/records"))
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--work-size", type=int, default=128)
    parser.add_argument("--samples-per-class", type=int, default=200)
    parser.add_argument("--max-classes", type=int, default=0, help="Use only the first N characters. Keep 0 for the official assignment.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index-only", action="store_true", help="Count classes and images only; do not write images.")
    args = parser.parse_args()

    if args.samples_per_class < 200 and args.max_classes == 0:
        print("Warning: the assignment requires at least 200 training samples per character.")
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

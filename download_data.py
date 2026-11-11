"""Download the Traditional Chinese Handwriting Dataset.

The assignment needs the full 13,065-character archive. The 4,803-character
common-word archive is only a pipeline check: its softmax would not have
13,065 outputs.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import class_and_file, decode_zip_filename

FULL_BASES = [
    "https://github.com/chenkenanalytic/handwritting_data_all/raw/master/",
    "https://media.githubusercontent.com/media/chenkenanalytic/handwritting_data_all/master/",
]
FULL_PARTS = [
    ("all_data.zip.001", 99_614_720),
    ("all_data.zip.002", 99_614_720),
    ("all_data.zip.003", 99_614_720),
    ("all_data.zip.004", 99_614_720),
    ("all_data.zip.005", 99_614_720),
    ("all_data.zip.006", 99_614_720),
    ("all_data.zip.007", 99_614_720),
    ("all_data.zip.008", 99_614_720),
    ("all_data.zip.009", 99_614_720),
    ("all_data.zip.010", 99_614_720),
    ("all_data.zip.011", 99_614_720),
    ("all_data.zip.012", 44_338_542),
]

COMMON_BASES = [
    "https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset/raw/master/data/",
    "https://media.githubusercontent.com/media/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset/master/data/",
]
COMMON_PARTS = [
    ("cleaned_data(50_50)-20200420T071507Z-001.zip", 38_525_034),
    ("cleaned_data(50_50)-20200420T071507Z-002.zip", 37_465_432),
    ("cleaned_data(50_50)-20200420T071507Z-003.zip", 37_248_813),
    ("cleaned_data(50_50)-20200420T071507Z-004.zip", 30_277_495),
]


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def download_file(bases: list, filename: str, dest: Path, expected: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == expected:
        print(f"略過，已下載 {dest.name}")
        return
    if dest.exists():
        dest.unlink()

    last_error = None
    for base in bases:
        url = base + quote(filename)
        print(f"下載 {filename}")
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=120) as response, dest.open("wb") as handle:
                downloaded = 0
                next_report = 20 * 1024 * 1024
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_report:
                        print(f"  {filename}: {downloaded / 1024 / 1024:.0f} MB")
                        next_report += 20 * 1024 * 1024
            if dest.stat().st_size != expected:
                raise IOError(f"大小是 {dest.stat().st_size}，預期 {expected}")
            print(f"完成 {filename}")
            return
        except (URLError, TimeoutError, OSError) as error:
            last_error = error
            print(f"  失敗：{error}")
            if dest.exists():
                dest.unlink()
    raise RuntimeError(f"無法下載 {filename}：{last_error}")


def concat_parts(parts: list, destination: Path) -> None:
    expected = sum(size for _, size in parts)
    if destination.exists() and destination.stat().st_size == expected:
        print(f"略過合併，已有 {destination.name}")
        return
    print(f"合併成分割壓縮檔 {destination.name}")
    with destination.open("wb") as output:
        for filename, _size in parts:
            part_path = destination.parent / filename
            with part_path.open("rb") as incoming:
                shutil.copyfileobj(incoming, output, length=1024 * 1024)
    if destination.stat().st_size != expected:
        raise IOError("合併後的壓縮檔大小不正確。")


def extract_common(zip_paths: list, output_dir: Path) -> None:
    """Turn the flat common-word archive into one folder per character."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for zip_path in zip_paths:
        print(f"整理 {zip_path.name}")
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                parsed = class_and_file(decode_zip_filename(info))
                if parsed is None:
                    continue
                character, filename = parsed
                target = output_dir / character / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.write_bytes(archive.read(info))
                    written += 1
    print(f"常用字圖片已放到 {output_dir}（新寫入 {written} 張）。這不是 13,065 字的完整資料集。")


def main() -> int:
    configure_stdio()
    parser = argparse.ArgumentParser(description="Download the handwriting dataset.")
    parser.add_argument("--variant", choices=("full", "common"), default="full")
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--keep-parts", action="store_true")
    args = parser.parse_args()

    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    if args.variant == "full":
        part_paths = []
        for filename, size in FULL_PARTS:
            path = output / filename
            download_file(FULL_BASES, filename, path, size)
            part_paths.append(path)
        archive = output / "all_data.zip"
        concat_parts(FULL_PARTS, archive)
        with zipfile.ZipFile(archive) as zipped:
            print(f"壓縮檔可開啟，成員數 {len(zipped.infolist())}。")
        if not args.keep_parts:
            for path in part_paths:
                path.unlink(missing_ok=True)
            print("已刪除分割檔，保留 data/all_data.zip。")
        print("下一步：python prepare_records.py --zip data/all_data.zip --out data/records")
        return 0

    part_paths = []
    for filename, size in COMMON_PARTS:
        path = output / filename
        download_file(COMMON_BASES, filename, path, size)
        part_paths.append(path)
    extract_common(part_paths, output / "common_cleaned")
    print("下一步（僅供流程測試）：python train.py --data-dir data/common_cleaned --max-classes 20 --epochs 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

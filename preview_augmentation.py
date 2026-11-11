"""Save a preview of the five training versions of one character."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.augment import TRANSFORMS, render_sample
from src.data import iter_first_image, read_gray_file


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main() -> int:
    configure_stdio()
    parser = argparse.ArgumentParser(description="Preview rotation, shear, and scaling.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--zip", type=Path)
    source.add_argument("--data-dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/augmentation_preview.png"))
    parser.add_argument("--img-size", type=int, default=128)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    installed = {font.name for font in font_manager.fontManager.ttflist}
    for family in ("Microsoft JhengHei", "Microsoft YaHei", "Noto Sans CJK TC", "Noto Sans CJK JP", "SimHei"):
        if family in installed:
            plt.rcParams["font.sans-serif"] = [family]
            plt.rcParams["axes.unicode_minus"] = False
            break
    else:
        family = ""

    if args.image:
        image = read_gray_file(args.image)
        title = args.image.name if family else "augmentation preview"
    else:
        character, filename, image = iter_first_image(args.zip, args.data_dir)
        title = f"{character} / {filename}" if family else "augmentation preview"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 5, figsize=(12, 3))
    for axis, aug_id in zip(axes, range(5)):
        preview = render_sample(image, aug_id, args.img_size, args.img_size)
        axis.imshow(preview, cmap="gray", vmin=0, vmax=255)
        axis.set_title(TRANSFORMS[aug_id]["name"], fontsize=9)
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(args.out, dpi=140)
    plt.close(figure)
    print(f"已儲存 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

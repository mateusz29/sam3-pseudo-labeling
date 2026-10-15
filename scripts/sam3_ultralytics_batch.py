"""Process a folder of images with Ultralytics SAM3."""

import argparse
from pathlib import Path

from sam3_ultralytics import create_predictor, process_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("sam3_results"), type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--conf", type=float, default=0.7)
    args = parser.parse_args()

    images = [p for p in args.input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    predictor = create_predictor(args.conf)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(images):
        output_path = args.output_dir / f"{image_path.stem}_sam3.jpg"
        label_path = output_path.with_suffix(".txt")
        count = process_image(predictor, image_path, output_path, args.prompt)
        print(f"{image_path.name}: {count} object(s) -> {output_path}, {label_path}")


if __name__ == "__main__":
    main()

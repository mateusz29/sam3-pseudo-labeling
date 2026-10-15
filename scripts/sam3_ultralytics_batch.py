"""Process a folder of images with Ultralytics SAM3."""

import argparse
from pathlib import Path

from ultralytics.models.sam import SAM3SemanticPredictor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("sam3_results"), type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    images = [
        p for p in args.input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    predictor = SAM3SemanticPredictor(
        overrides={
            "model": "sam3.pt",
            "task": "segment",
            "mode": "predict",
            "conf": args.conf,
            "verbose": False,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(images):
        predictor.set_image(str(image_path))
        result = predictor(text=[args.prompt])[0]
        output_path = args.output_dir / f"{image_path.stem}_sam3.jpg"
        result.save(filename=str(output_path))
        print(
            f"{image_path.name}: {0 if result.boxes is None else len(result.boxes)} object(s) -> {output_path}"
        )


if __name__ == "__main__":
    main()

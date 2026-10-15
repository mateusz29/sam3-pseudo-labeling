"""Find text-prompted concepts in one image with Ultralytics SAM3."""

import argparse
from pathlib import Path

from ultralytics.models.sam import SAM3SemanticPredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--prompt", required=True, help="For example: airplane or ship")
    parser.add_argument("--output", type=Path, default=Path("sam3_result.jpg"))
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    predictor = SAM3SemanticPredictor(
        overrides={
            "model": "sam3.pt",
            "task": "segment",
            "mode": "predict",
            "conf": args.conf,
            "verbose": False,
        }
    )
    predictor.set_image(str(args.image))
    result = predictor(text=[args.prompt])[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(filename=str(args.output))
    count = 0 if result.boxes is None else len(result.boxes)
    print(f"Found {count} '{args.prompt}' object(s). Saved {args.output}")


if __name__ == "__main__":
    main()

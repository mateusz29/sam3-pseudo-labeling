"""Find text-prompted concepts in one image with Ultralytics SAM3."""

import argparse
from pathlib import Path

from PIL import Image
from sam3_output import save_boxes
from ultralytics.models.sam import SAM3SemanticPredictor


def create_predictor(conf: float) -> SAM3SemanticPredictor:
    return SAM3SemanticPredictor(
        overrides={
            "model": "sam3.pt",
            "task": "segment",
            "mode": "predict",
            "conf": conf,
            "verbose": False,
        }
    )


def process_image(
    predictor: SAM3SemanticPredictor,
    image_path: Path,
    output_path: Path,
    prompt: str,
) -> int:
    predictor.set_image(str(image_path))
    result = predictor(text=[prompt])[0]
    boxes = [] if result.boxes is None else result.boxes.xyxy.cpu().tolist()
    scores = [] if result.boxes is None else result.boxes.conf.cpu().tolist()
    label_path = output_path.with_suffix(".txt")
    return save_boxes(
        Image.open(image_path), boxes, scores, output_path, label_path, prompt
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--prompt", required=True, help="For example: airplane or ship")
    parser.add_argument("--output", type=Path, default=Path("sam3_result.jpg"))
    parser.add_argument("--conf", type=float, default=0.7)
    args = parser.parse_args()

    predictor = create_predictor(args.conf)
    count = process_image(predictor, args.image, args.output, args.prompt)
    label_path = args.output.with_suffix(".txt")
    print(f"Found {count} '{args.prompt}' object(s). Saved {args.output} and {label_path}")


if __name__ == "__main__":
    main()

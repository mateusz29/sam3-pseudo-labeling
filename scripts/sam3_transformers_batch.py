"""Process a folder of images with Hugging Face Transformers SAM3."""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import Sam3Model, Sam3Processor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("sam3_results"), type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(args.input_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, text=args.prompt, return_tensors="pt").to(
            device
        )
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_instance_segmentation(
            outputs,
            threshold=args.conf,
            mask_threshold=0.5,
            target_sizes=inputs["original_sizes"].tolist(),
        )[0]
        boxes = result["boxes"].detach().cpu().tolist()
        scores = result["scores"].detach().cpu().tolist()
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = [round(v) for v in box]
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            draw.text((x1, max(0, y1 - 16)), f"{args.prompt} {score:.2f}", fill="red")
        output_path = args.output_dir / f"{image_path.stem}_sam3.jpg"
        canvas.save(output_path)
        print(f"{image_path.name}: {len(boxes)} object(s) -> {output_path}")


if __name__ == "__main__":
    main()

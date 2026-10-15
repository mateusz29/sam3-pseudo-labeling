"""Shared drawing and YOLO-label output helpers."""

from collections.abc import Iterable, Sequence
from pathlib import Path

from PIL import Image, ImageDraw


def save_boxes(
    image: Image.Image,
    boxes: Iterable[Sequence[float]],
    scores: Iterable[float],
    output_image: Path,
    output_labels: Path,
    prompt: str,
) -> int:
    image = image.convert("RGB")
    width, height = image.size
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    lines = []
    for box, score in zip(boxes, scores, strict=False):
        x1, y1, x2, y2 = [float(value) for value in box]
        x1, x2 = sorted((max(0.0, min(width, x1)), max(0.0, min(width, x2))))
        y1, y2 = sorted((max(0.0, min(height, y1)), max(0.0, min(height, y2))))
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1, max(0, y1 - 16)), f"{prompt} {float(score):.2f}", fill="red")
        lines.append(
            f"0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
        )
    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_labels.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_image)
    output_labels.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)

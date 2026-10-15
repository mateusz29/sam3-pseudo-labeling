# SAM3-to-YOLO Dataset Study

Research code for studying whether SAM3-generated bounding boxes can help train YOLO detectors on aerial and satellite imagery.

The planned comparison is:

1. Train YOLO on an existing manually annotated dataset.
2. Use SAM3 text prompts to generate bounding boxes for an additional image dataset.
3. Train YOLO on the original dataset plus the SAM3-generated annotations.
4. Compare both models on the same held-out evaluation set.

The repository currently contains simple SAM3 inference scripts for generating annotated images. Dataset conversion, YOLO training, and experiment tracking can be added around these scripts.

## Generate annotated images

```powershell
uv run python scripts/sam3_ultralytics.py `
  --image path\to\image.jpg `
  --prompt airplane `
  --output results\image_airplane.jpg
```

For a folder:

```powershell
uv run python scripts/sam3_ultralytics_batch.py `
  --input-dir images `
  --output-dir results `
  --prompt airplane
```

The scripts save annotated copies containing only bounding boxes, labels, and confidence scores. They also save normalized YOLO label files beside the images. Original images are not overwritten by default. Useful prompts include `airplane`, `ship`, `vehicle`, and `building`.

## Setup

Install the environment with `uv`:

```powershell
uv sync --extra ultralytics --group dev
```

On Windows x64, `torch` and `torchvision` use the CUDA 12.8 PyTorch index. On other platforms, `uv` uses the normal PyPI packages automatically. This is a platform rule, not GPU detection; if another Windows computer needs CPU-only PyTorch, adjust the CUDA source entries in `pyproject.toml` locally.

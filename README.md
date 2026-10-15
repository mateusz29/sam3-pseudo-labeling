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
uv run python sam3_ultralytics.py `
  --image path\to\image.jpg `
  --prompt airplane `
  --output results\image_airplane.jpg
```

For a folder:

```powershell
uv run python sam3_ultralytics_batch.py `
  --input-dir images `
  --output-dir results `
  --prompt airplane
```

The scripts save annotated copies containing bounding boxes, labels, confidence scores, and, for Ultralytics, rendered masks. Original images are not overwritten by default. Useful prompts include `airplane`, `ship`, `vehicle`, `building`.

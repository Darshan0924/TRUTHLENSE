# Training the TruthLens baseline

TruthLens includes a small, three-class PyTorch CNN, training and evaluation
commands, and Grad-CAM explanations. This page documents the custom three-class
training workflow. For ready-made binary AI-image screening, use the separate
[pretrained setup](PRETRAINED.md). A working training pipeline is not evidence of
useful detection accuracy; neither workflow provides a forensic guarantee.

Provide legally usable, labeled image data from representative sources:

```text
dataset/
  train/
    authentic/
    ai_generated/
    manipulated/
  val/
    authentic/
    ai_generated/
    manipulated/
  test/
    authentic/
    ai_generated/
    manipulated/
```

Every class folder needs at least one readable JPG, PNG, WebP, BMP, or TIFF image.
Files can be nested in these folders. Label authentic media captured without
content synthesis or manipulation; label wholly generated media `ai_generated`;
label edited, composited, or face-swapped real media `manipulated`. Establish a
consistent labeling policy for ambiguous mixtures before collecting data.

Split data by source, identity, scene, and generator where appropriate before
training; frames from a single video must stay in one split. The loader rejects
identical normalized images appearing in different splits, including differently
named or losslessly re-encoded copies, and rejects identical images assigned
conflicting labels even in the same split. It cannot automatically detect all near
duplicates, shared identities, or related video frames. Validation data selects
the checkpoint; test data measures the selected model once.

From the repository root:

```bash
uv run --frozen python scripts/train.py --dataset /path/to/dataset --output models/truthlens.pt --epochs 10
uv run --frozen python scripts/evaluate.py --dataset /path/to/dataset --checkpoint models/truthlens.pt --output models/test-metrics.json
```

The default CPU baseline uses RGB input, EXIF orientation, transparency composited
over white consistently for training and uploads, a bilinear resize to
128 by 128, and channel values normalized from `[0, 255]` to `[-1, 1]`.
Its three convolutions have 16, 32, and 64 channels, followed by global average
pooling and a linear classifier. Training uses cross-entropy, Adam, seed 42,
two CPU threads, and no external model downloads. `--seed`, `--threads`,
`--learning-rate`, `--batch-size`, and `--epochs` are configurable. Given the same
data and environment, deterministic CPU training reproduces the weights.

The lowest validation-loss checkpoint is saved atomically. It includes model
weights, architecture identifier, class order, image size, seed, split counts,
training timestamp, and validation metrics. Evaluation reports test accuracy,
per-class precision, recall, F1 and support, macro averages, cross-entropy loss,
and a confusion matrix whose rows are actual labels and columns predicted
labels. Reports are JSON and are suitable for an experiment log.

Set `TRUTHLENS_MODEL_PATH` to the trained checkpoint's absolute path before
starting the server. The checkpoint loader checks architecture, class ordering,
weight shapes, finite weights, and metadata, and uses PyTorch's `weights_only`
loading. Use only checkpoints you created or obtained from a trusted source.

Grad-CAM maps show relative attention for the predicted class, from low attention
(blue) to high attention (red). Evidence images preserve the oriented aspect ratio
and have at most 1200 pixels on the longest side to bound memory usage; this does
not change the classifier's 128px preprocessing. They do not establish that highlighted pixels
were edited, locate tampering reliably, or prove a media item's provenance.
Softmax confidence is a model score, not a calibrated probability of truth.
Video analysis applies this image model to sampled frames and aggregates their
scores; it has no trained temporal model or audio analysis.

The integration test trains one epoch on nine synthetic fixture images to verify
checkpoint round-tripping, inference, Grad-CAM, held-out evaluation, and leakage
rejection. Those images are test fixtures, not a dataset or a useful detector.
Before drawing conclusions, evaluate on a sufficiently large, representative,
separate dataset, report failures and domain shifts, and consider calibration.

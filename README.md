# TruthLens AI

A working foundation for the TruthLens AI project: upload an image or video, inspect sampled frames and image quality, and run a pretrained AI-image detector or a locally trained PyTorch classifier. Built with Python, FastAPI, OpenCV, PyTorch, and a responsive HTML/CSS/JavaScript dashboard.

**A pretrained detector can now be downloaded with one setup command.** It screens for AI-generated imagery using two classes and does not detect face swaps or general manipulation. The separate custom training workflow retains `authentic`, `ai_generated`, and `manipulated` classes. Scores are uncalibrated estimates, not proof of authenticity. No dataset or large model weights are stored in Git.

## Run locally

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/). The lockfile selects CPU PyTorch so a GPU is not required.

```bash
uv sync --frozen --python 3.12
uv run --frozen python scripts/setup_pretrained.py
uv run --frozen uvicorn truthlens.app:app --host 127.0.0.1 --port 8000
```

The pretrained setup downloads approximately **348 MB**, verifies the publisher's weight checksum, and stores a pinned local model under `models/pretrained/`. Internet access is needed for setup; later inference stays on your computer. The app automatically loads this verified model on startup. If the app was already running, stop it with **Ctrl+C** and restart it after model setup. An upload made before loading the model keeps its original report; upload again for new predictions.

The default model is [Organika's SDXL detector](https://huggingface.co/Organika/sdxl-detector), published under **CC BY-NC 3.0** for noncommercial use with attribution. It was fine-tuned from [umm-maybe's AI art detector](https://huggingface.co/umm-maybe/AI-image-detector) on Wikimedia/SDXL pairs. This project uses it for an educational demonstration. See [pretrained setup and limitations](docs/PRETRAINED.md) before interpreting a score or planning another use.

Open the local service in a browser. The dashboard supports image/video upload, frame inspection, quality diagnostics, saved analysis history, and JSON report downloads. Preview frames and reports are stored locally under `.truthlens/`; uploaded originals are deleted after processing. No uploads are sent to a remote AI service.

For development with reload:

```bash
uv run --frozen uvicorn truthlens.app:app --reload --host 127.0.0.1 --port 8000
```

Use `TRUTHLENS_DATA_DIR` to choose another storage directory, `TRUTHLENS_PRETRAINED_DIR` to choose a verified pretrained bundle, and `TRUTHLENS_MODEL_PATH` to load a trusted custom trained checkpoint. A custom checkpoint takes priority over the pretrained model. Set these variables before starting the server. A missing or invalid model is shown as unavailable; media inspection remains usable.

## Train and evaluate

Follow [the model workflow](docs/ML.md) for the supported dataset layout, training command, evaluation, checkpoint format, and attention-map limitations. Use separate train, validation, and test splits. Keep frames from the same original video, subject, or generation source in one split to prevent leakage. Use a recognized licensed deepfake/synthetic-media dataset; preserve the source labels rather than guessing whether an unlabeled file is real.

No dataset download is automated because access, consent, licensing, and class labels vary by source. For example, FaceForensics++ can support authentic/manipulated data under its terms; the AI-generated class also needs an appropriately labeled source. Do not describe a model trained on toy fixtures as a useful authenticity detector.

## Verify

```bash
uv run --frozen --extra dev pytest -q
```

Tests exercise actual image and video processing, upload rejection, stored reports, training, checkpoint loading, inference, and evaluation. Tiny generated test fixtures check that the software works; they do not measure real-world detection performance.

## Architecture

```text
Browser dashboard
  → FastAPI upload validation
  → Pillow image decoding / OpenCV video sampling
  → quality measurements + frontal face regions + frame thumbnails
  → pretrained binary image classifier OR custom three-class CNN + Grad-CAM
  → mean frame probabilities for video prediction
  → SQLite history + downloadable JSON evidence report
```

The pretrained model is a Swin image classifier. The custom training model is a compact three-class CNN baseline. Video analysis samples a bounded number of frames; it is image-model aggregation, not a temporally trained video forensic model. OpenCV's frontal-face cascade identifies optional regions for inspection; the classifiers use the full frame, and face detection can miss faces. Image brightness, contrast, and sharpness are quality measurements, not evidence that media is synthetic. Grad-CAM is available for the custom CNN and highlights what influenced it; it does not identify proven tampered regions. No heatmap is fabricated for the pretrained transformer.

## Scope and next steps

This first version implements the proposal's web interface, preprocessing, video sampling, learned-model integration, training, validation/test metrics, and evidence reports. The remaining research work is to curate licensed data, train the baseline, evaluate on unseen sources, calibrate confidence, and compare stronger pretrained architectures. Audio detection, user accounts, and hosted deployment are outside this version.

The service is intended for a local development environment. It has no user authentication or tenant isolation. Bind to loopback by default; add authentication, TLS, retention controls, and resource isolation before exposing it publicly. History and media remain in the local data directory until you remove them while the server is stopped.

See [the API reference](docs/API.md) for endpoints, response semantics, and processing limits.

The project requirements were extracted from the supplied TruthLens AI approval presentation. Its example confidence figure was an illustration, not a trained model result.

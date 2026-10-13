# Pretrained AI-image screening

The default pretrained model is **Organika/sdxl-detector**, by Organika, fine-tuned from **umm-maybe/AI-image-detector**. Model source and full card: https://huggingface.co/Organika/sdxl-detector.

Model revision: `b37fede8562cb72b89ec201c0987f96ba21b518a`.

License: **CC BY-NC 3.0**, attribution required, noncommercial use. The model card describes personal/educational use and the training-data background. Large weights are downloaded separately; the source repository contains no model weights. The downloaded bundle retains the original model card, license metadata, source revision, and hashes. Keep this attribution when sharing the project.

## Setup in VS Code on Windows

Open a terminal in the folder containing `pyproject.toml`. Stop an existing TruthLens server with Ctrl+C, then run each command after the preceding one finishes:

```powershell
uv sync --frozen --python 3.12
uv run --frozen python scripts/setup_pretrained.py
uv run --frozen uvicorn truthlens.app:app --port 8000
```

Setup downloads approximately 348 MB. The application automatically loads `models/pretrained/manifest.json` and its verified files. A subsequent setup run verifies the existing bundle without downloading it again. Inference uses CPU and works without internet after setup. Upload media again after restarting; previous reports preserve their original model state.

If an incomplete folder already contains files without a valid manifest, setup refuses to overwrite them. Select a new empty folder with `--output PATH`, then set `TRUTHLENS_PRETRAINED_DIR` to that folder before starting. An explicitly configured `TRUTHLENS_MODEL_PATH` takes priority and loads your custom three-class CNN instead.

## Meaning of the output

The source classes are `artificial` and `human`; TruthLens maps them to `ai_generated` and `authentic` internally. The dashboard says **Likely AI-generated** and **Likely human-created** to avoid treating the latter as a forensic authenticity finding. Only those two probabilities appear. No separate manipulation score is generated.

The model was fine-tuned on Wikimedia/SDXL pairs. Results may differ for other generators, compressed or resized images, screenshots, and other sources. Publisher-reported validation metrics are not TruthLens's own measured accuracy. No accuracy claim is made for your data or current generators.

For videos, the app applies the image classifier to at most 12 sampled frames and averages the probabilities. It does not have a temporal deepfake detector, audio detector, or face-swap classifier. The pretrained transformer provides no Grad-CAM overlay in this integration; the custom CNN workflow still supports its own attention maps.

## Integrity and troubleshooting

The downloader resolves the pinned revision, checks the public model card and class mappings, and verifies the publisher's LFS SHA256 and file size. This default uses safetensors directly. A compatible legacy `.bin` model can be converted using strictly `torch.load(weights_only=True)` after checksum verification; arbitrary model code or unsafe pickle fallback is never enabled. Runtime verifies the saved bundle and loads locally with `trust_remote_code=False`, `use_safetensors=True`, and `local_files_only=True`.

TLS verification remains enabled. Required destinations include `huggingface.co` and its actual model CDN, currently `us.aws.cdn.hf.co`; network policy errors need the destination enabled in environment settings. No token is required for the default public model. Do not disable TLS or checksum verification to resolve an error.

If startup says **Model could not load**, rerun setup in the correct project folder and inspect its error. Refresh dependencies with `uv sync --frozen --python 3.12`, verify the bundle is readable, then restart. Setup failures are not successful model installation.

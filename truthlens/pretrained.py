"""A verified, local-only binary pretrained image detector.

The public model distinguishes human-created and AI-generated images. It does
not classify face swaps or general manipulation, and its scores are uncalibrated.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from truthlens.ml import ModelLoadError, normalize_image

DEFAULT_MODEL_ID = "Organika/sdxl-detector"
DEFAULT_MODEL_DIRECTORY = Path("models/pretrained")
MANIFEST_NAME = "manifest.json"
REQUIRED_FILES = ("config.json", "preprocessor_config.json", "model.safetensors", "README.md")
CLASSES = ("authentic", "ai_generated")
SOURCE_LABELS = {"human": "authentic", "artificial": "ai_generated"}
SUPPORTED_ARCHITECTURES = {
    "swin": "SwinForImageClassification",
    "vit": "ViTForImageClassification",
}
SUPPORTED_LICENSES = {
    "apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "cc-by-3.0", "cc-by-4.0", "cc0-1.0", "cc-by-nc-3.0"
}
LIMITATIONS = [
    "Screens for AI-generated images; does not detect face swaps or general image manipulation.",
    "A human-created prediction is not proof of authenticity.",
    "Scores are uncalibrated model probabilities, not verified certainty; inspect results independently.",
    "Accuracy on your images and on sampled video frames has not been evaluated.",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_mapping(config: dict[str, Any]) -> dict[int, str]:
    """Reject unfamiliar class orders instead of guessing the positive class."""
    raw = config.get("id2label")
    if not isinstance(raw, dict) or set(raw) != {"0", "1"}:
        raise ValueError("The pretrained model must define exactly two id2label entries (0 and 1).")
    labels = {}
    for index, label in raw.items():
        if not isinstance(label, str) or label.strip().lower() not in SOURCE_LABELS:
            raise ValueError("Expected the pretrained source labels 'artificial' and 'human'.")
        labels[int(index)] = SOURCE_LABELS[label.strip().lower()]
    if set(labels.values()) != set(CLASSES):
        raise ValueError("The pretrained model must contain both artificial and human classes.")
    if config.get("num_labels", 2) != 2:
        raise ValueError("The pretrained model must have exactly two output classes.")
    label2id = config.get("label2id")
    if label2id is not None:
        if not isinstance(label2id, dict) or len(label2id) != 2:
            raise ValueError("Invalid pretrained label2id mapping.")
        for label, index in label2id.items():
            # The published model uses string IDs in label2id. Accept only
            # these exact integer representations, never guess label order.
            if isinstance(index, str) and index in {"0", "1"}:
                index = int(index)
            if not isinstance(label, str) or type(index) is not int:
                raise ValueError("Invalid pretrained label2id mapping.")
            if labels.get(index) != SOURCE_LABELS.get(label.strip().lower()):
                raise ValueError("The pretrained label2id and id2label mappings disagree.")
    return labels


def validate_config(config: dict[str, Any]) -> dict[int, str]:
    architecture = SUPPORTED_ARCHITECTURES.get(config.get("model_type"))
    if architecture is None or config.get("architectures") != [architecture]:
        raise ValueError("Only standard Swin/ViT image-classification architectures are supported.")
    if config.get("auto_map"):
        raise ValueError("Custom remote model code is not supported.")
    return label_mapping(config)


def verify_bundle(directory: Path) -> tuple[dict[str, Any], dict[str, Any], dict[int, str]]:
    """Verify the immutable source metadata and each local file before loading."""
    directory = Path(directory)
    manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
        raise ValueError("Unsupported pretrained manifest format.")
    model_id, revision = manifest.get("model_id"), manifest.get("revision")
    if not isinstance(model_id, str) or not re.fullmatch(r"[\w.-]+/[\w.-]+", model_id):
        raise ValueError("The pretrained manifest is missing a valid source model ID.")
    if not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("The pretrained source revision must be an immutable commit SHA.")
    if manifest.get("license") not in SUPPORTED_LICENSES:
        raise ValueError("The model license is unknown or unsupported; review its public model card.")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(REQUIRED_FILES):
        raise ValueError("The pretrained manifest must list the four required model files.")
    for filename in REQUIRED_FILES:
        expected = files[filename]
        if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
            raise ValueError(f"Missing valid SHA256 for pretrained file {filename}.")
        path = directory / filename
        if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Pretrained file {filename} is missing or failed its SHA256 verification.")
    source_weights = manifest.get("weights_source")
    if not isinstance(source_weights, dict):
        raise ValueError("The pretrained manifest is missing its authoritative weights source.")
    source_name = source_weights.get("filename")
    source_sha = source_weights.get("lfs_sha256")
    if (source_name not in {"model.safetensors", "pytorch_model.bin"}
            or not isinstance(source_sha, str) or not re.fullmatch(r"[a-f0-9]{64}", source_sha)
            or source_weights.get("verified_sha256") != source_sha):
        raise ValueError("The pretrained weights source has no verified authoritative LFS SHA256.")
    if source_name == "model.safetensors":
        if source_sha != files["model.safetensors"] or manifest.get("conversion") is not None:
            raise ValueError("The pretrained weights do not match their authoritative LFS SHA256.")
    elif manifest.get("conversion") != "torch.load(weights_only=True)->safetensors":
        raise ValueError("The pretrained weights require a recorded safe safetensors conversion.")
    if manifest.get("source_url") != f"https://huggingface.co/{model_id}/tree/{revision}":
        raise ValueError("The pretrained manifest has an inconsistent source URL.")
    if manifest.get("capabilities") != ["ai_generation"]:
        raise ValueError("Unsupported pretrained model capabilities.")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Invalid pretrained model configuration.")
    labels = validate_config(config)
    if manifest.get("label_mapping") != {str(index): label for index, label in labels.items()}:
        raise ValueError("The pretrained manifest and model class mappings disagree.")
    processor_config = json.loads((directory / "preprocessor_config.json").read_text(encoding="utf-8"))
    if not isinstance(processor_config, dict) or processor_config.get("auto_map"):
        raise ValueError("Custom remote image processor code is not supported.")
    return manifest, config, labels


class PretrainedDetector:
    """Load only verified safetensors, then use CPU inference without networking."""

    def __init__(self, directory: Path):
        self._model = None
        try:
            manifest, config, self._labels = verify_bundle(directory)
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            torch.set_num_threads(min(2, torch.get_num_threads()))
            self._processor = AutoImageProcessor.from_pretrained(
                str(directory), local_files_only=True, trust_remote_code=False, use_fast=False
            )
            model, loading = AutoModelForImageClassification.from_pretrained(
                str(directory), local_files_only=True, trust_remote_code=False,
                use_safetensors=True, output_loading_info=True,
            )
            if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                raise ValueError("Pretrained weights do not exactly match the declared architecture.")
            if model.config.model_type != config["model_type"] or model.config.num_labels != 2:
                raise ValueError("Loaded model does not match the verified binary model configuration.")
            if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
                raise ValueError("Pretrained weights contain non-finite values.")
            self._model = model.to("cpu").eval()
            self._metadata = {
                "model_id": manifest["model_id"], "revision": manifest["revision"],
                "license": manifest["license"], "source_url": manifest["source_url"],
                "architecture": config["architectures"][0],
                "license_notice": (
                    "CC BY-NC 3.0: attribution required; personal and educational noncommercial use only."
                    if manifest["license"] == "cc-by-nc-3.0" else "See the source model card for attribution and license terms."
                ),
            }
        except (ImportError, OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            raise ModelLoadError(f"Cannot load pretrained detector: {exc}") from exc

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def info(self) -> dict[str, Any]:
        limitations = list(LIMITATIONS)
        if self._metadata["model_id"] == "Organika/sdxl-detector":
            limitations.append("This detector was fine-tuned on SDXL images; performance can differ for other generators and image sources.")
        return {
            "ready": self.ready, "status": "ready" if self.ready else "unconfigured",
            "classes": list(CLASSES), "capabilities": ["ai_generation"],
            "heatmap_available": False, "limitations": limitations,
            **self._metadata,
        }

    def predict(self, image: Image.Image) -> dict[str, Any]:
        import torch

        with torch.inference_mode():
            inputs = self._processor(images=normalize_image(image), return_tensors="pt")
            inputs = {name: value.to("cpu") for name, value in inputs.items()}
            logits = self._model(**inputs).logits
            if tuple(logits.shape) != (1, 2) or not torch.isfinite(logits).all():
                raise ValueError("Pretrained inference returned invalid binary logits.")
            values = logits.softmax(dim=1)[0].detach().cpu().tolist()
        probabilities = {self._labels[index]: float(value) for index, value in enumerate(values)}
        winner = max(probabilities, key=probabilities.get)
        return {"label": winner, "confidence": probabilities[winner], "probabilities": probabilities}

    def predict_with_heatmap(self, image: Image.Image) -> tuple[dict[str, Any], None]:
        # No unvalidated transformer explanation is passed off as CNN Grad-CAM.
        return self.predict(image), None

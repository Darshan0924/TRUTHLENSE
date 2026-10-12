"""Trainable baseline detector and Grad-CAM evidence for TruthLens.

No classifier weights are downloaded or fabricated. An unconfigured Detector
cannot classify an image. Its scores require evaluation on representative data.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from pickle import UnpicklingError
from typing import Any

from PIL import Image, ImageOps

CLASSES = ("authentic", "ai_generated", "manipulated")
ARCHITECTURE = "truthlens_tiny_cnn_v1"
IMAGE_SIZE = 128
HEATMAP_MAX_SIZE = 1200
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class ModelNotReadyError(RuntimeError):
    """Inference was requested before a trained checkpoint was configured."""


class ModelLoadError(ValueError):
    """A local checkpoint is missing, incompatible, or malformed."""


class DatasetError(ValueError):
    """The dataset cannot support a valid train/validation/test workflow."""


def build_model():
    """Create the small CPU-friendly CNN without downloading external weights."""
    import torch.nn as nn

    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(64, len(CLASSES))

        def logits_from_features(self, features):
            return self.classifier(self.pool(features).flatten(1))

        def forward(self, images):
            return self.logits_from_features(self.features(images))

    return TinyCNN()


def normalize_image(image: Image.Image) -> Image.Image:
    """Apply the same orientation and white alpha composite to data and uploads."""
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def preprocess(image: Image.Image):
    """Normalize to RGB, resize to 128px and map [0,255] to [-1,1]."""
    import numpy as np
    import torch

    resized = normalize_image(image).resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
    )
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy()).sub_(0.5).div_(0.5)


class Detector:
    """Load a user-trained, compatible checkpoint and run CPU inference."""

    def __init__(self, checkpoint: Path | None = None):
        self._model = None
        self._metadata: dict[str, Any] = {}
        if checkpoint is None:
            return
        path = Path(checkpoint)
        if not path.is_file():
            raise ModelLoadError(f"Checkpoint file does not exist: {path.name}")
        try:
            import torch

            # Many cloud CPUs advertise far more threads than this small model
            # can use; cap startup inference to avoid thread scheduling overhead.
            torch.set_num_threads(min(2, torch.get_num_threads()))
            # weights_only prevents loading arbitrary Python checkpoint objects.
            checkpoint_data = torch.load(path, map_location="cpu", weights_only=True)
            self._validate_checkpoint(checkpoint_data)
            model = build_model()
            model.load_state_dict(checkpoint_data["state_dict"], strict=True)
            if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
                raise ValueError("Checkpoint weights contain non-finite values")
            model.eval()
            self._metadata = {
                "training": checkpoint_data.get("training", {}),
                "metrics": checkpoint_data.get("metrics", {}),
            }
            self._model = model
        except (ImportError, OSError, RuntimeError, ValueError, TypeError, KeyError, EOFError, UnpicklingError) as exc:
            raise ModelLoadError(f"Cannot load TruthLens checkpoint: {exc}") from exc

    @staticmethod
    def _validate_checkpoint(data: Any) -> None:
        if not isinstance(data, dict):
            raise ValueError("Checkpoint must be a mapping")
        if data.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format version")
        if data.get("architecture") != ARCHITECTURE:
            raise ValueError("Incompatible checkpoint architecture")
        if data.get("classes") != list(CLASSES):
            raise ValueError("Checkpoint classes must match the three TruthLens classes")
        if data.get("image_size") != IMAGE_SIZE:
            raise ValueError("Checkpoint image size must be 128")
        if not isinstance(data.get("state_dict"), dict):
            raise ValueError("Checkpoint is missing model weights")
        # Metadata is shown by the API; insist on JSON-safe, finite values.
        for key in ("training", "metrics"):
            if not isinstance(data.get(key, {}), dict):
                raise ValueError(f"Checkpoint {key} metadata must be a mapping")
            json.dumps(data.get(key, {}), allow_nan=False)

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def info(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "status": "ready" if self.ready else "unconfigured",
            "architecture": ARCHITECTURE,
            "classes": list(CLASSES),
            "image_size": IMAGE_SIZE,
            **self._metadata,
        }

    def _require_model(self):
        if self._model is None:
            raise ModelNotReadyError("Train and configure a checkpoint before analysis.")
        return self._model

    @staticmethod
    def _prediction(logits) -> dict[str, Any]:
        probabilities = logits.softmax(dim=1)[0].detach().cpu().tolist()
        winner = max(range(len(CLASSES)), key=lambda index: probabilities[index])
        return {
            "label": CLASSES[winner],
            "probabilities": dict(zip(CLASSES, probabilities)),
            "confidence": probabilities[winner],
        }

    def predict(self, image: Image.Image) -> dict[str, Any]:
        model = self._require_model()
        import torch

        with torch.inference_mode():
            return self._prediction(model(preprocess(image).unsqueeze(0)))

    def predict_with_heatmap(self, image: Image.Image) -> tuple[dict[str, Any], Image.Image]:
        """Return a prediction and Grad-CAM heatmap of relative class attention.

        The RGB heatmap follows the EXIF-oriented aspect ratio, with longest side
        at most 1200px. It explains relative model attention, not verified edits.
        """
        model = self._require_model()
        import numpy as np
        import torch

        # Calculate input gradients without accumulating or mutating model grads.
        # This also permits independent concurrent requests to share the model.
        with torch.enable_grad():
            tensor = preprocess(image).unsqueeze(0).requires_grad_(True)
            features = model.features(tensor)
            logits = model.logits_from_features(features)
            prediction = self._prediction(logits)
            target = CLASSES.index(prediction["label"])
            gradients = torch.autograd.grad(logits[0, target], features)[0]
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * features).sum(dim=1)[0].relu().detach().numpy()
        peak = float(cam.max())
        if peak > 0:
            cam = cam / peak
        oriented = ImageOps.exif_transpose(image)
        # Bound colorization allocations for high-resolution uploaded media.
        # Only evidence dimensions change; classifier input remains unchanged.
        oriented.thumbnail((HEATMAP_MAX_SIZE, HEATMAP_MAX_SIZE))
        grayscale = Image.fromarray((cam * 255).astype(np.uint8)).resize(
            oriented.size, Image.Resampling.BILINEAR
        )
        intensity = np.asarray(grayscale, dtype=np.float32) / 255.0
        # Blue -> cyan -> yellow -> red; intensity, not likelihood of tampering.
        red = np.clip(1.5 - np.abs(4.0 * intensity - 3.0), 0.0, 1.0)
        green = np.clip(1.5 - np.abs(4.0 * intensity - 2.0), 0.0, 1.0)
        blue = np.clip(1.5 - np.abs(4.0 * intensity - 1.0), 0.0, 1.0)
        heatmap = Image.fromarray((np.stack((red, green, blue), axis=-1) * 255).astype(np.uint8))
        return prediction, heatmap


def classification_metrics(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    """Calculate multiclass metrics; confusion matrix rows are actual classes."""
    if not labels or len(labels) != len(predictions):
        raise ValueError("Expected nonempty, equally sized labels and predictions")
    confusion = [[0 for _ in CLASSES] for _ in CLASSES]
    for actual, predicted in zip(labels, predictions):
        if type(actual) is not int or type(predicted) is not int:
            raise ValueError("Labels and predictions must be integer class indices")
        if not (0 <= actual < len(CLASSES) and 0 <= predicted < len(CLASSES)):
            raise ValueError("Class index outside the supported class range")
        confusion[actual][predicted] += 1
    per_class = {}
    for index, class_name in enumerate(CLASSES):
        true_positives = confusion[index][index]
        predicted_count = sum(row[index] for row in confusion)
        support = sum(confusion[index])
        precision = true_positives / predicted_count if predicted_count else 0.0
        recall = true_positives / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    macro = {
        key: sum(per_class[name][key] for name in CLASSES) / len(CLASSES)
        for key in ("precision", "recall", "f1")
    }
    return {
        "accuracy": sum(confusion[index][index] for index in range(len(CLASSES))) / len(labels),
        "macro": macro,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "class_order": list(CLASSES),
        "samples": len(labels),
    }


def inspect_dataset(root: Path) -> dict[str, list[tuple[Path, int]]]:
    """Validate every class/split and reject decoded-pixel duplication across splits."""
    root = Path(root)
    samples: dict[str, list[tuple[Path, int]]] = {}
    seen: dict[bytes, tuple[str, Path, int]] = {}
    for split in ("train", "val", "test"):
        samples[split] = []
        for label, class_name in enumerate(CLASSES):
            folder = root / split / class_name
            if not folder.is_dir():
                raise DatasetError(f"Missing dataset folder: {folder}")
            paths = sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
            if not paths:
                raise DatasetError(f"No supported images in {folder}")
            for path in paths:
                try:
                    with Image.open(path) as image:
                        pixels = normalize_image(image)
                        digest = hashlib.sha256()
                        digest.update(f"{pixels.width}x{pixels.height}:".encode())
                        digest.update(pixels.tobytes())
                        fingerprint = digest.digest()
                except (OSError, ValueError, Image.DecompressionBombError) as exc:
                    raise DatasetError(f"Cannot decode image {path}: {exc}") from exc
                previous = seen.get(fingerprint)
                if previous is not None and previous[2] != label:
                    raise DatasetError(
                        f"Contradictory labels: identical decoded pixels labeled {CLASSES[previous[2]]} ({previous[1]}) "
                        f"and {class_name} ({path})"
                    )
                if previous is not None and previous[0] != split:
                    raise DatasetError(
                        f"Data leakage: identical decoded pixels in {previous[0]} ({previous[1]}) and {split} ({path})"
                    )
                # Keep the first split; later cross-split duplicates must fail.
                seen.setdefault(fingerprint, (split, path, label))
                samples[split].append((path, label))
    return samples


def _loader(samples, *, batch_size: int, shuffle: bool = False, seed: int = 42):
    import torch
    from torch.utils.data import DataLoader, Dataset

    class ImageDataset(Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, index):
            path, label = samples[index]
            with Image.open(path) as image:
                return preprocess(image), label

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ImageDataset(), batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator)


def _run_evaluation(model, loader) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    labels, predictions = [], []
    total_loss = 0.0
    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images)
            total_loss += functional.cross_entropy(logits, targets, reduction="sum").item()
            labels.extend(targets.tolist())
            predictions.extend(logits.argmax(dim=1).tolist())
    metrics = classification_metrics(labels, predictions)
    metrics["loss"] = total_loss / len(labels)
    return metrics


def train_model(
    dataset: Path,
    output: Path,
    *,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    seed: int = 42,
    threads: int = 2,
    progress=None,
) -> dict[str, Any]:
    """Train on train/, select on val/, and leave test/ unused by optimization."""
    if epochs < 1 or batch_size < 1 or threads < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Epochs, batch size, threads and learning rate must be positive")
    samples = inspect_dataset(dataset)
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = _loader(samples["train"], batch_size=batch_size, shuffle=True, seed=seed)
    val_loader = _loader(samples["val"], batch_size=batch_size)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_metrics = {}
    best_epoch = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        training_loss = 0.0
        correct = 0
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.item() * len(targets)
            correct += int((logits.argmax(dim=1) == targets).sum())
        validation = _run_evaluation(model, val_loader)
        row = {
            "epoch": epoch,
            "train_loss": training_loss / len(samples["train"]),
            "train_accuracy": correct / len(samples["train"]),
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro"]["f1"],
        }
        if not all(math.isfinite(value) for value in row.values()):
            raise ValueError("Training diverged: non-finite loss or metric")
        history.append(row)
        if progress is not None:
            progress(row)
        if validation["loss"] < best_loss:
            best_loss, best_metrics, best_epoch = validation["loss"], validation, epoch
            checkpoint = {
                "format_version": 1,
                "architecture": ARCHITECTURE,
                "classes": list(CLASSES),
                "image_size": IMAGE_SIZE,
                "state_dict": model.state_dict(),
                "metrics": {"validation": validation},
                "training": {
                    "epoch": epoch,
                    "epochs_requested": epochs,
                    "seed": seed,
                    "learning_rate": learning_rate,
                    "split_samples": {key: len(value) for key, value in samples.items()},
                    "trained_at": datetime.now(timezone.utc).isoformat(),
                },
            }
            temporary = output.with_name(output.name + ".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(output)
    return {"checkpoint": str(output), "best_epoch": best_epoch, "validation": best_metrics, "history": history}


def evaluate_model(checkpoint: Path, dataset: Path, *, batch_size: int = 16, threads: int = 2) -> dict[str, Any]:
    """Evaluate the untouched test split and report classification metrics."""
    if batch_size < 1 or threads < 1:
        raise ValueError("Batch size and threads must be positive")
    samples = inspect_dataset(dataset)
    detector = Detector(checkpoint)
    import torch

    torch.set_num_threads(threads)
    metrics = _run_evaluation(detector._require_model(), _loader(samples["test"], batch_size=batch_size))
    return {"split": "test", "checkpoint": str(checkpoint), **metrics}

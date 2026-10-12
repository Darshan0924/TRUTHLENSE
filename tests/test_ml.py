from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from truthlens.ml import (
    CLASSES,
    DatasetError,
    Detector,
    ModelLoadError,
    ModelNotReadyError,
    classification_metrics,
    evaluate_model,
    inspect_dataset,
    normalize_image,
    preprocess,
    train_model,
)


def make_dataset(root: Path):
    """Tiny distinct images test the pipeline, not real detection quality."""
    for split_index, split in enumerate(("train", "val", "test")):
        for class_index, name in enumerate(CLASSES):
            folder = root / split / name
            folder.mkdir(parents=True)
            pixels = np.zeros((20, 24, 3), dtype=np.uint8)
            pixels[:, :, class_index] = 150 + split_index * 20
            pixels[split_index : split_index + 3, :, :] = 10 + class_index * 20
            Image.fromarray(pixels).save(folder / "sample.png")


def test_unconfigured_detector_never_invents_a_result():
    detector = Detector()
    assert detector.ready is False
    assert detector.info["status"] == "unconfigured"
    with pytest.raises(ModelNotReadyError):
        detector.predict(Image.new("RGB", (8, 8)))
    with pytest.raises(ModelNotReadyError):
        detector.predict_with_heatmap(Image.new("RGB", (8, 8)))


def test_metrics_distinguish_false_positives_and_false_negatives():
    metrics = classification_metrics([0, 0, 1, 2], [0, 1, 1, 1])
    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [0, 1, 0]]
    assert metrics["accuracy"] == 0.5
    assert metrics["per_class"]["authentic"]["precision"] == 1.0
    assert metrics["per_class"]["authentic"]["recall"] == 0.5
    assert metrics["per_class"]["ai_generated"]["precision"] == pytest.approx(1 / 3)
    assert metrics["per_class"]["manipulated"]["f1"] == 0.0
    assert metrics["macro"]["f1"] == pytest.approx((2 / 3 + 0.5) / 3)


@pytest.mark.parametrize("labels,predictions", [([], []), ([0], []), ([3], [0]), ([0.5], [0])])
def test_metrics_reject_invalid_labels(labels, predictions):
    with pytest.raises(ValueError):
        classification_metrics(labels, predictions)


def test_dataset_rejects_cross_split_pixel_leakage(tmp_path):
    make_dataset(tmp_path)
    original = tmp_path / "train" / "authentic" / "sample.png"
    # Different filename and encoding still expose the same decoded image.
    with Image.open(original) as image:
        image.save(tmp_path / "test" / "authentic" / "duplicate.bmp")
    with pytest.raises(DatasetError, match="Data leakage"):
        inspect_dataset(tmp_path)


def test_dataset_requires_all_classes_in_each_split(tmp_path):
    make_dataset(tmp_path)
    (tmp_path / "val" / "manipulated" / "sample.png").unlink()
    with pytest.raises(DatasetError, match="No supported images"):
        inspect_dataset(tmp_path)


def test_dataset_rejects_contradictory_labels_within_one_split(tmp_path):
    make_dataset(tmp_path)
    original = tmp_path / "train" / "authentic" / "sample.png"
    with Image.open(original) as image:
        image.save(tmp_path / "train" / "manipulated" / "contradictory.png")
    with pytest.raises(DatasetError, match="Contradictory labels"):
        inspect_dataset(tmp_path)


@pytest.mark.parametrize("mode", ["RGBA", "LA", "P"])
def test_transparent_inputs_match_white_composited_training_data(mode):
    import torch

    if mode == "RGBA":
        transparent = Image.new(mode, (11, 7), (40, 70, 100, 0))
        transparent.putpixel((2, 3), (100, 30, 90, 120))
    elif mode == "LA":
        transparent = Image.new(mode, (11, 7), (80, 0))
        transparent.putpixel((2, 3), (100, 120))
    else:
        transparent = Image.new(mode, (11, 7), 0)
        transparent.putpalette([40, 70, 100, 100, 30, 90] + [0] * 762)
        transparent.info["transparency"] = 0
        transparent.putpixel((2, 3), 1)
    rgba = transparent.convert("RGBA")
    composited = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
    normalized = normalize_image(transparent)
    assert normalized.mode == "RGB"
    assert normalized.tobytes() == composited.tobytes()
    assert normalized.getpixel((0, 0)) == (255, 255, 255)
    assert torch.equal(preprocess(transparent), preprocess(composited))


def test_dataset_rejects_transparent_and_composited_cross_split_duplicates(tmp_path):
    make_dataset(tmp_path)
    transparent = Image.new("RGBA", (11, 7), (30, 70, 90, 0))
    transparent.putpixel((2, 3), (100, 30, 90, 120))
    transparent.save(tmp_path / "train" / "authentic" / "transparent.png")
    normalize_image(transparent).save(tmp_path / "test" / "authentic" / "composited.png")
    with pytest.raises(DatasetError, match="Data leakage"):
        inspect_dataset(tmp_path)


def test_corrupt_checkpoint_has_a_useful_error(tmp_path):
    checkpoint = tmp_path / "corrupt.pt"
    checkpoint.write_bytes(b"not a torch checkpoint")
    with pytest.raises(ModelLoadError, match="Cannot load"):
        Detector(checkpoint)


def test_training_inference_gradcam_and_held_out_evaluation(tmp_path):
    import torch

    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "model.pt"
    make_dataset(dataset)
    result = train_model(dataset, checkpoint, epochs=1, batch_size=3, seed=7, threads=1)
    assert checkpoint.is_file()
    assert result["best_epoch"] == 1
    assert result["validation"]["samples"] == 3
    assert len(result["history"]) == 1
    detector = Detector(checkpoint)
    assert detector.ready
    assert detector.info["training"]["split_samples"] == {"train": 3, "val": 3, "test": 3}
    image = Image.new("RGB", (37, 21), (30, 80, 130))
    prediction = detector.predict(image)
    explanation, heatmap = detector.predict_with_heatmap(image)
    assert prediction["label"] in CLASSES
    assert set(prediction["probabilities"]) == set(CLASSES)
    assert sum(prediction["probabilities"].values()) == pytest.approx(1.0)
    assert prediction["confidence"] == max(prediction["probabilities"].values())
    assert explanation["probabilities"] == pytest.approx(prediction["probabilities"])
    assert heatmap.size == image.size
    assert heatmap.mode == "RGB"
    large_image = Image.new("RGB", (1800, 1400), (30, 80, 130))
    _, large_heatmap = detector.predict_with_heatmap(large_image)
    assert max(large_heatmap.size) <= 1200
    assert large_heatmap.width / large_heatmap.height == pytest.approx(1800 / 1400, abs=0.002)
    assert large_image.size == (1800, 1400)
    metrics = evaluate_model(checkpoint, dataset, batch_size=3, threads=1)
    assert metrics["split"] == "test"
    assert metrics["samples"] == 3
    assert sum(map(sum, metrics["confusion_matrix"])) == 3
    assert set(metrics["macro"]) == {"precision", "recall", "f1"}
    # Incompatible labels cannot silently reinterpret checkpoint outputs.
    payload = torch.load(checkpoint, weights_only=True)
    payload["classes"] = ["real", "fake"]
    torch.save(payload, tmp_path / "incompatible.pt")
    with pytest.raises(ModelLoadError, match="classes"):
        Detector(tmp_path / "incompatible.pt")

"""Offline integration and integrity checks; tiny fixture weights prove no accuracy."""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from scripts import setup_pretrained as setup
from truthlens.ml import ModelLoadError
from truthlens.pretrained import (
    DEFAULT_MODEL_ID,
    MANIFEST_NAME,
    REQUIRED_FILES,
    PretrainedDetector,
    file_sha256,
    label_mapping,
    verify_bundle,
)


REVISION = setup.DEFAULT_REVISION


def write_manifest(directory: Path, *, source_name="model.safetensors", source_sha=None):
    config = json.loads((directory / "config.json").read_text())
    source_sha = source_sha or file_sha256(directory / "model.safetensors")
    manifest = {
        "format_version": 1,
        "model_id": DEFAULT_MODEL_ID,
        "revision": REVISION,
        "license": "cc-by-4.0",
        "source_url": f"https://huggingface.co/{DEFAULT_MODEL_ID}/tree/{REVISION}",
        "capabilities": ["ai_generation"],
        "label_mapping": {str(key): value for key, value in label_mapping(config).items()},
        "files": {name: file_sha256(directory / name) for name in REQUIRED_FILES},
        "weights_source": {
            "filename": source_name,
            "lfs_sha256": source_sha,
            "verified_sha256": source_sha,
        },
        "conversion": None if source_name == "model.safetensors" else "torch.load(weights_only=True)->safetensors",
    }
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest))
    return manifest


@pytest.fixture(scope="session")
def tiny_source(tmp_path_factory):
    """A real local Swin model with synthetic weights, never a performance benchmark."""
    from transformers import SwinConfig, SwinForImageClassification, ViTImageProcessor

    directory = tmp_path_factory.mktemp("tiny-swin")
    config = SwinConfig(
        image_size=32,
        patch_size=4,
        embed_dim=8,
        depths=[1, 1],
        num_heads=[1, 2],
        window_size=4,
        num_labels=2,
        id2label={0: "human", 1: "artificial"},
        label2id={"human": 0, "artificial": 1},
    )
    with torch.random.fork_rng():
        torch.manual_seed(7)
        model = SwinForImageClassification(config)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([3.0, -3.0]))
    model.save_pretrained(directory, safe_serialization=True)
    ViTImageProcessor(size={"height": 32, "width": 32}).save_pretrained(directory)
    (directory / "README.md").write_text("---\nlicense: cc-by-4.0\n---\nSynthetic test fixture.\n")
    write_manifest(directory)
    return directory


@pytest.fixture
def bundle(tmp_path, tiny_source):
    directory = tmp_path / "bundle"
    shutil.copytree(tiny_source, directory)
    return directory


def change_json(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def mock_registry(monkeypatch, source, *, weight_name="model.safetensors", metadata_change=None):
    """Mock registry and files; any accidental additional network operation fails."""
    import huggingface_hub

    weight_path = source / weight_name
    info = SimpleNamespace(
        sha=REVISION,
        private=False,
        gated=False,
        card_data={"license": "cc-by-4.0"},
        siblings=[
            SimpleNamespace(rfilename=name, lfs=None)
            for name in ("config.json", "preprocessor_config.json", "README.md")
        ] + [SimpleNamespace(
            rfilename=weight_name,
            lfs={"sha256": file_sha256(weight_path), "size": weight_path.stat().st_size},
        )],
    )
    if metadata_change:
        metadata_change(info)
    calls = []

    class Api:
        def model_info(self, model_id, **kwargs):
            calls.append(("info", model_id, kwargs))
            return info

    def download(model_id, filename, **kwargs):
        assert model_id == DEFAULT_MODEL_ID
        assert kwargs["revision"] == REVISION
        assert "cache_dir" in kwargs
        calls.append(("download", filename, kwargs))
        return str(source / filename)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return calls


@pytest.mark.parametrize("labels,expected", [
    ({"0": "human", "1": "artificial"}, {0: "authentic", 1: "ai_generated"}),
    ({"0": "artificial", "1": "human"}, {0: "ai_generated", 1: "authentic"}),
])
def test_source_label_order_is_explicit(labels, expected):
    assert label_mapping({"id2label": labels}) == expected


def test_published_string_label_ids_match_integer_ids():
    labels = {"0": "artificial", "1": "human"}
    integers = {"id2label": labels, "label2id": {"artificial": 0, "human": 1}}
    strings = {"id2label": labels, "label2id": {"artificial": "0", "human": "1"}}
    assert label_mapping(strings) == label_mapping(integers) == {0: "ai_generated", 1: "authentic"}


@pytest.mark.parametrize("invalid_index", [True, False, 0.0, "00", "1.0", "human", "2", None])
def test_label_ids_reject_bool_and_noncanonical_strings(invalid_index):
    config = {
        "id2label": {"0": "artificial", "1": "human"},
        "label2id": {"artificial": invalid_index, "human": "1"},
    }
    with pytest.raises(ValueError):
        label_mapping(config)


@pytest.mark.parametrize("config", [
    {"id2label": {"0": "LABEL_0", "1": "LABEL_1"}},
    {"id2label": {"0": "human", "1": "human"}},
    {"id2label": {"1": "human", "2": "artificial"}},
    {"id2label": {"0": "human", "1": "artificial"}, "num_labels": 3},
    {"id2label": {"0": "human", "1": "artificial"}, "label2id": {"human": 1, "artificial": 0}},
])
def test_ambiguous_or_inconsistent_labels_fail(config):
    with pytest.raises(ValueError):
        label_mapping(config)


def test_real_swin_loader_is_strictly_offline_and_remaps_scores(bundle, monkeypatch):
    import socket
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor_loader = AutoImageProcessor.from_pretrained
    model_loader = AutoModelForImageClassification.from_pretrained
    calls = {}

    def forbidden(*args, **kwargs):
        pytest.fail("A verified local bundle must not access the network")

    def processor(*args, **kwargs):
        calls["processor"] = kwargs
        return processor_loader(*args, **kwargs)

    def model(*args, **kwargs):
        calls["model"] = kwargs
        return model_loader(*args, **kwargs)

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(AutoImageProcessor, "from_pretrained", processor)
    monkeypatch.setattr(AutoModelForImageClassification, "from_pretrained", model)
    detector = PretrainedDetector(bundle)
    for kwargs in calls.values():
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False
    assert calls["model"]["use_safetensors"] is True
    assert calls["model"]["output_loading_info"] is True
    prediction, heatmap = detector.predict_with_heatmap(Image.new("RGBA", (19, 23), (100, 60, 80, 0)))
    assert prediction["label"] == "authentic"
    assert prediction["probabilities"]["authentic"] == pytest.approx(0.9975274)
    assert sum(prediction["probabilities"].values()) == pytest.approx(1.0)
    assert set(prediction["probabilities"]) == {"authentic", "ai_generated"}
    assert heatmap is None
    assert detector.info["heatmap_available"] is False
    assert detector.info["capabilities"] == ["ai_generation"]
    assert detector.info["revision"] == REVISION


def test_reversed_source_order_changes_semantics_not_class_index(bundle):
    def flip(config):
        config["id2label"] = {"0": "artificial", "1": "human"}
        config["label2id"] = {"artificial": 0, "human": 1}
    change_json(bundle / "config.json", flip)
    write_manifest(bundle)
    prediction = PretrainedDetector(bundle).predict(Image.new("RGB", (32, 32)))
    assert prediction["label"] == "ai_generated"
    assert prediction["probabilities"]["ai_generated"] > 0.99


@pytest.mark.parametrize("filename", REQUIRED_FILES)
def test_every_bundle_file_is_hash_verified_before_loading(bundle, filename, monkeypatch):
    from transformers import AutoModelForImageClassification

    def forbidden(*args, **kwargs):
        pytest.fail("Corrupt files must be rejected before model loading")

    monkeypatch.setattr(AutoModelForImageClassification, "from_pretrained", forbidden)
    with (bundle / filename).open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ModelLoadError, match="SHA256"):
        PretrainedDetector(bundle)


@pytest.mark.parametrize("field,value", [
    ("revision", "main"),
    ("license", "unknown"),
    ("source_url", "https://example.com/model"),
    ("capabilities", ["face_swap"]),
    ("label_mapping", {"0": "ai_generated", "1": "authentic"}),
    ("conversion", "torch.load(weights_only=False)->safetensors"),
])
def test_inconsistent_manifest_is_rejected(bundle, field, value):
    change_json(bundle / MANIFEST_NAME, lambda manifest: manifest.update({field: value}))
    with pytest.raises(ValueError):
        verify_bundle(bundle)


def test_authoritative_weights_sha_is_required(bundle):
    change_json(bundle / MANIFEST_NAME, lambda value: value["weights_source"].update({"verified_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="authoritative LFS SHA256"):
        verify_bundle(bundle)


@pytest.mark.parametrize("filename", ["config.json", "preprocessor_config.json"])
def test_remote_custom_code_is_rejected_even_with_valid_hashes(bundle, filename):
    change_json(bundle / filename, lambda value: value.update({"auto_map": {"AutoModel": "evil.module"}}))
    write_manifest(bundle)
    with pytest.raises(ModelLoadError, match="remote"):
        PretrainedDetector(bundle)


def test_missing_weights_never_produce_a_randomly_initialized_detector(bundle):
    weights = load_file(bundle / "model.safetensors")
    del weights["classifier.weight"]
    save_file(weights, bundle / "model.safetensors")
    write_manifest(bundle)
    with pytest.raises(ModelLoadError, match="exactly match"):
        PretrainedDetector(bundle)


def test_nonfinite_weights_are_rejected(bundle):
    weights = load_file(bundle / "model.safetensors")
    weights["classifier.bias"][0] = float("nan")
    save_file(weights, bundle / "model.safetensors")
    write_manifest(bundle)
    with pytest.raises(ModelLoadError, match="non-finite"):
        PretrainedDetector(bundle)


@pytest.mark.parametrize("logits", [
    torch.tensor([[float("nan"), 0.0]]),
    torch.tensor([[float("inf"), 0.0]]),
    torch.zeros(1, 3),
    torch.zeros(2, 2),
])
def test_invalid_inference_logits_never_become_predictions(bundle, monkeypatch, logits):
    detector = PretrainedDetector(bundle)
    monkeypatch.setattr(detector._model, "forward", lambda **kwargs: SimpleNamespace(logits=logits))
    with pytest.raises(ValueError, match="binary logits"):
        detector.predict(Image.new("RGB", (32, 32)))


def test_verified_legacy_state_dict_converts_with_no_pickle_fallback(tmp_path, monkeypatch):
    source, output = tmp_path / "weights.bin", tmp_path / "model.safetensors"
    expected = {"classifier.weight": torch.tensor([[1.0, 2.0]]), "classifier.bias": torch.tensor([3.0])}
    torch.save(expected, source)
    original_load = torch.load
    calls = []

    def tracked(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", tracked)
    setup._convert_verified_state_dict(source, output)
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    actual = load_file(output)
    assert set(actual) == set(expected)
    for name in expected:
        assert torch.equal(actual[name], expected[name])


def test_unsupported_pickle_rejection_does_not_retry_unsafe_loading(tmp_path, monkeypatch):
    calls = []

    def reject(*args, **kwargs):
        calls.append(kwargs)
        raise pickle.UnpicklingError("Unsupported object in weights_only input")

    monkeypatch.setattr(torch, "load", reject)
    output = tmp_path / "model.safetensors"
    with pytest.raises((pickle.UnpicklingError, ValueError), match="Unsupported|unsupported"):
        setup._convert_verified_state_dict(tmp_path / "weights.bin", output)
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    assert not output.exists()


@pytest.mark.parametrize("state", [
    {},
    {"metadata": "not a tensor"},
    {"weight": torch.tensor([float("nan")])},
    {"weight": torch.sparse_coo_tensor([[0]], [1.0], (2,))},
])
def test_conversion_rejects_invalid_state_dicts(tmp_path, state):
    source, output = tmp_path / "weights.bin", tmp_path / "model.safetensors"
    torch.save(state, source)
    with pytest.raises(ValueError):
        setup._convert_verified_state_dict(source, output)
    assert not output.exists()


def test_installer_pins_downloads_and_is_idempotent_offline(tmp_path, tiny_source, monkeypatch):
    calls = mock_registry(monkeypatch, tiny_source)
    output = tmp_path / "installed"
    result = setup.install_pretrained(output, revision="main")
    assert result["revision"] == REVISION
    assert len([call for call in calls if call[0] == "download"]) == 4
    assert calls[0] == ("info", DEFAULT_MODEL_ID, {"revision": "main", "files_metadata": True})
    assert not list(tmp_path.glob(".truthlens-pretrained-*"))
    hashes = {name: file_sha256(output / name) for name in (*REQUIRED_FILES, MANIFEST_NAME)}

    def no_network(*args, **kwargs):
        pytest.fail("An existing verified bundle must be reusable offline")

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", no_network)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_network)
    assert setup.install_pretrained(output) == result
    assert hashes == {name: file_sha256(output / name) for name in hashes}
    with pytest.raises(ValueError, match="different revision"):
        setup.install_pretrained(output, revision="a" * 40)


def test_bin_installer_verifies_source_then_records_safe_conversion(tmp_path, bundle, monkeypatch):
    torch.save(load_file(bundle / "model.safetensors"), bundle / "pytorch_model.bin")
    mock_registry(monkeypatch, bundle, weight_name="pytorch_model.bin")
    output = tmp_path / "installed"
    setup.install_pretrained(output)
    manifest, _, _ = verify_bundle(output)
    assert manifest["conversion"] == "torch.load(weights_only=True)->safetensors"
    assert manifest["weights_source"]["lfs_sha256"] == file_sha256(bundle / "pytorch_model.bin")
    assert not (output / "pytorch_model.bin").exists()
    assert PretrainedDetector(output).predict(Image.new("RGB", (16, 16)))["label"] == "authentic"


def test_failed_registry_integrity_never_publishes_partial_bundle(tmp_path, tiny_source, monkeypatch):
    def corrupt_metadata(info):
        info.siblings[-1].lfs["sha256"] = "0" * 64
    mock_registry(monkeypatch, tiny_source, metadata_change=corrupt_metadata)
    output = tmp_path / "installed"
    with pytest.raises(ValueError, match="LFS SHA256/size"):
        setup.install_pretrained(output)
    assert not output.exists()
    assert not list(tmp_path.glob(".truthlens-pretrained-*"))


def test_failed_bin_hash_verification_never_invokes_pickle_loader(tmp_path, bundle, monkeypatch):
    torch.save(load_file(bundle / "model.safetensors"), bundle / "pytorch_model.bin")

    def corrupt_metadata(info):
        info.siblings[-1].lfs["sha256"] = "0" * 64

    mock_registry(monkeypatch, bundle, weight_name="pytorch_model.bin", metadata_change=corrupt_metadata)

    def forbidden(*args, **kwargs):
        pytest.fail("Unverified source weights must never reach torch.load")

    monkeypatch.setattr(torch, "load", forbidden)
    output = tmp_path / "installed"
    with pytest.raises(ValueError, match="LFS SHA256/size"):
        setup.install_pretrained(output)
    assert not output.exists()
    assert not list(tmp_path.glob(".truthlens-pretrained-*"))


def test_cli_download_errors_do_not_print_signed_urls(monkeypatch, capsys):
    import sys
    from requests import HTTPError

    def fail(*args, **kwargs):
        raise HTTPError("403 for https://example.com/weights?signature=private-download-token")

    monkeypatch.setattr(setup, "install_pretrained", fail)
    monkeypatch.setattr(sys, "argv", ["setup_pretrained.py"])
    with pytest.raises(SystemExit) as exc:
        setup.main()
    assert exc.value.code == 1
    stderr = capsys.readouterr().err
    assert "HTTPError" in stderr
    assert "signature" not in stderr
    assert "private-download-token" not in stderr


def test_failed_strict_loading_cleans_temporary_downloads(tmp_path, bundle, monkeypatch):
    weights = load_file(bundle / "model.safetensors")
    del weights["classifier.bias"]
    save_file(weights, bundle / "model.safetensors")
    mock_registry(monkeypatch, bundle)
    output = tmp_path / "installed"
    output.mkdir()
    with pytest.raises(ModelLoadError, match="exactly match"):
        setup.install_pretrained(output)
    assert output.is_dir() and not list(output.iterdir())
    assert not list(tmp_path.glob(".truthlens-pretrained-*"))


def test_installer_preserves_unverified_user_files(tmp_path):
    output = tmp_path / "installed"
    output.mkdir()
    local_file = output / "notes.txt"
    local_file.write_text("keep this user file")
    with pytest.raises(ValueError, match="already contains files"):
        setup.install_pretrained(output)
    assert local_file.read_text() == "keep this user file"

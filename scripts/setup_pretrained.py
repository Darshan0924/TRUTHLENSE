#!/usr/bin/env python3
"""Download and verify a public binary AI-image detector for local CPU inference.

Run from the project folder: uv run --frozen python scripts/setup_pretrained.py
Weights stay outside Git. Downloads retain TLS and LFS SHA256 verification. No
custom model code executes; legacy PyTorch state dictionaries use weights_only.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import sys
import tempfile
from pickle import UnpicklingError
from pathlib import Path
from typing import Any

from requests import RequestException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from truthlens.pretrained import (
    DEFAULT_MODEL_DIRECTORY, DEFAULT_MODEL_ID, MANIFEST_NAME, REQUIRED_FILES,
    SUPPORTED_LICENSES, PretrainedDetector, file_sha256, validate_config,
)

MAX_WEIGHT_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_REVISION = "b37fede8562cb72b89ec201c0987f96ba21b518a"

# Use the supported HTTPS download transport. Native Xet otherwise uses a
# global cache that may be read-only in cloud development environments.
# TLS and the authoritative weight checksum remain verified below.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def _field(value: Any, name: str, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _license(card: Any) -> str:
    value = _field(card, "license")
    if not isinstance(value, str) or value.lower().strip() not in SUPPORTED_LICENSES:
        raise ValueError(
            "The public model card has an unknown or unsupported license. "
            "Use a model with an explicit supported license; do not infer one from its code repository."
        )
    return value.lower().strip()


def _convert_verified_state_dict(source: Path, destination: Path) -> None:
    """Never unpickle arbitrary objects or retry with weights_only=False."""
    import torch
    from safetensors.torch import save_file

    torch.set_num_threads(min(2, torch.get_num_threads()))
    state_dict = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Expected a nonempty tensor state dictionary in the verified PyTorch weights.")
    safe_tensors = {}
    for name, value in state_dict.items():
        if not isinstance(name, str) or type(value) is not torch.Tensor:
            raise ValueError("The verified PyTorch weights contain unsupported non-tensor data.")
        if value.layout != torch.strided or not torch.isfinite(value).all():
            raise ValueError("The verified PyTorch weights contain unsupported or non-finite tensors.")
        safe_tensors[name] = value.contiguous().clone()
    save_file(safe_tensors, str(destination))


def install_pretrained(
    directory: Path = DEFAULT_MODEL_DIRECTORY,
    *, model_id: str = DEFAULT_MODEL_ID, revision: str = DEFAULT_REVISION,
) -> dict[str, Any]:
    """Resolve a ref once, download at its SHA, and atomically publish a verified bundle."""
    directory = Path(directory)
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", model_id):
        raise ValueError("Specify a public Hugging Face model ID in owner/model format.")
    if (directory / MANIFEST_NAME).is_file():
        detector = PretrainedDetector(directory)
        if detector.info["model_id"] != model_id:
            raise ValueError("An existing verified bundle uses a different model; choose another output folder.")
        if re.fullmatch(r"[a-f0-9]{40}", revision) and detector.info["revision"] != revision:
            raise ValueError("An existing verified bundle uses a different revision; choose another output folder.")
        return detector.info
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError("The output folder already contains files without a verified manifest; choose an empty folder.")

    from huggingface_hub import HfApi, ModelCard, hf_hub_download
    from huggingface_hub.utils import logging as hub_logging

    # Retry diagnostics can contain signed, temporary CDN query parameters.
    # Keep those out of CLI output; the final error is handled below.
    hub_logging.set_verbosity_error()

    info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
    commit = info.sha
    if not isinstance(commit, str) or not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("The model registry did not provide a valid immutable source revision.")
    if _field(info, "private", False) or _field(info, "gated", False):
        raise ValueError("Choose an ungated public detector that does not require access credentials.")
    license_name = _license(_field(info, "card_data"))
    siblings = {_field(item, "rfilename"): item for item in info.siblings}
    metadata_files = ["config.json", "preprocessor_config.json", "README.md"]
    if any(name not in siblings for name in metadata_files):
        raise ValueError("The public model is missing its config, preprocessor config, or model card.")
    weight_name = next((name for name in ("model.safetensors", "pytorch_model.bin") if name in siblings), None)
    if weight_name is None:
        raise ValueError("The model has no supported single-file safetensors or PyTorch state dictionary.")
    weight = siblings[weight_name]
    lfs = _field(weight, "lfs")
    expected = _field(lfs, "sha256")
    size = _field(lfs, "size")
    if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
        raise ValueError("The model registry has no authoritative LFS SHA256 for its weights; download refused.")
    if type(size) is not int or size <= 0 or size > MAX_WEIGHT_BYTES:
        raise ValueError("The pretrained weights have an invalid size or exceed the 2 GiB setup limit.")

    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".truthlens-pretrained-", dir=directory.parent) as temp_name:
        temp = Path(temp_name)
        bundle = temp / "bundle"
        bundle.mkdir()
        cache = temp / "hub-cache"
        for filename in metadata_files:
            downloaded = hf_hub_download(model_id, filename, revision=commit, cache_dir=cache)
            shutil.copyfile(downloaded, bundle / filename)
        card = ModelCard.load(str(bundle / "README.md"))
        if _license(card.data) != license_name:
            raise ValueError("The downloaded model card and registry disagree on the model license.")
        config = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
        labels = validate_config(config)
        downloaded_weights = Path(hf_hub_download(model_id, weight_name, revision=commit, cache_dir=cache))
        if downloaded_weights.stat().st_size != size or file_sha256(downloaded_weights) != expected:
            raise ValueError("Downloaded pretrained weights failed authoritative LFS SHA256/size verification.")
        if weight_name == "model.safetensors":
            shutil.copyfile(downloaded_weights, bundle / "model.safetensors")
            conversion = None
        else:
            _convert_verified_state_dict(downloaded_weights, bundle / "model.safetensors")
            conversion = "torch.load(weights_only=True)->safetensors"
        manifest = {
            "format_version": 1, "model_id": model_id, "revision": commit, "license": license_name,
            "source_url": f"https://huggingface.co/{model_id}/tree/{commit}",
            "capabilities": ["ai_generation"],
            "label_mapping": {str(index): label for index, label in labels.items()},
            "files": {filename: file_sha256(bundle / filename) for filename in REQUIRED_FILES},
            "weights_source": {"filename": weight_name, "lfs_sha256": expected, "verified_sha256": expected},
            "conversion": conversion,
        }
        (bundle / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        # Verify both the complete bundle and strict architecture loading before publishing it.
        detector = PretrainedDetector(bundle)
        model_info = detector.info
        # Release memory-mapped weights before renaming the directory on Windows.
        del detector
        gc.collect()
        if directory.exists():
            directory.rmdir()  # Only the empty directory accepted above may be replaced.
        bundle.replace(directory)
        return model_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_DIRECTORY)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Immutable model revision; a different ref is resolved once and pinned")
    args = parser.parse_args()
    try:
        info = install_pretrained(args.output, model_id=args.model_id, revision=args.revision)
    except (ImportError, OSError, RuntimeError, ValueError, TypeError, KeyError, UnpicklingError, EOFError, RequestException) as exc:
        if isinstance(exc, RequestException):
            message = f"Model download failed ({type(exc).__name__}). Check internet access to huggingface.co and its model-download domains."
        else:
            message = str(exc)
        parser.exit(1, f"Pretrained setup failed: {message}\n")
    print(f"Pretrained detector ready: {info['model_id']} ({info['revision'][:12]}, {info['license']}).")
    print("Restart TruthLens and upload media again to get binary AI-image screening results.")
    print("This detector does not classify face swaps or general manipulation; scores are uncalibrated.")
    print(info["license_notice"])


if __name__ == "__main__":
    main()

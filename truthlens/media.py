"""Bounded media decoding and honest, model-dependent analysis."""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from truthlens.ml import normalize_image

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_PIXELS = 25_000_000
MAX_DIMENSION = 8192
MAX_VIDEO_SECONDS = 600
MAX_FRAMES = 12
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm"}
CLASSES = ("authentic", "ai_generated", "manipulated")


class MediaError(ValueError):
    """The upload cannot safely be decoded as supported media."""


class InferenceError(RuntimeError):
    """The configured model could not produce a valid prediction."""


def _check_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise MediaError("The file has invalid media dimensions.")
    if width > MAX_DIMENSION or height > MAX_DIMENSION or width * height > MAX_PIXELS:
        raise MediaError(
            f"Media must be at most {MAX_DIMENSION} pixels per side and {MAX_PIXELS:,} pixels in total."
        )


def _rgb(image: Image.Image) -> Image.Image:
    return normalize_image(image)


def _image_frames(path: Path) -> tuple[dict[str, Any], list[tuple[float, Image.Image]]]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    raise MediaError("Only JPEG, PNG, and WebP images are supported.")
                image_format = image.format
                _check_dimensions(*image.size)
                if getattr(image, "n_frames", 1) != 1:
                    raise MediaError("Animated images are not supported; upload a video instead.")
                image.verify()
            with Image.open(path) as image:
                image.load()
                frame = _rgb(image)
    except MediaError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise MediaError("The file is not a valid, safely decodable image.") from exc
    width, height = frame.size
    return (
        {"format": image_format, "width": width, "height": height,
         "duration_seconds": None, "frame_count": 1, "fps": None},
        [(0.0, frame)],
    )


def _video_signature(path: Path, extension: str) -> None:
    with path.open("rb") as stream:
        header = stream.read(16)
    valid = False
    if extension in {".mp4", ".mov"}:
        valid = len(header) >= 12 and header[4:8] in {b"ftyp", b"moov", b"mdat", b"wide", b"free"}
    elif extension == ".avi":
        valid = header[:4] == b"RIFF" and header[8:12] == b"AVI "
    elif extension == ".webm":
        valid = header[:4] == b"\x1a\x45\xdf\xa3"
    if not valid:
        raise MediaError("The file does not contain a supported video container.")


def _video_frames(path: Path, extension: str) -> tuple[dict[str, Any], Iterator[tuple[float, Image.Image]]]:
    _video_signature(path, extension)
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise MediaError("The video could not be decoded. Its codec may be unsupported.")
        width_raw = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height_raw = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count_raw = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not all(math.isfinite(value) for value in (width_raw, height_raw, fps, frame_count_raw)):
            raise MediaError("The video has invalid stream metadata.")
        width, height = int(width_raw), int(height_raw)
        _check_dimensions(width, height)
        frame_count = int(frame_count_raw)
        if fps <= 0 or fps > 240 or frame_count <= 0:
            raise MediaError("The video must have a readable frame count and a frame rate of at most 240 fps.")
        duration = frame_count / fps
        if duration > MAX_VIDEO_SECONDS:
            raise MediaError(f"Videos must be at most {MAX_VIDEO_SECONDS // 60} minutes long.")
        count = min(MAX_FRAMES, frame_count)
        indices = [round(index * (frame_count - 1) / (count - 1)) for index in range(count)] if count > 1 else [0]
        codec_value = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((codec_value >> (8 * shift)) & 0xFF) for shift in range(4))
        metadata = {"format": extension[1:].upper(), "width": width, "height": height,
                    "duration_seconds": round(duration, 4), "frame_count": frame_count,
                    "fps": round(fps, 4), "codec": codec if codec.isprintable() else "unknown"}
    except cv2.error as exc:
        capture.release()
        raise MediaError("The video could not be safely decoded.") from exc
    except BaseException:
        capture.release()
        raise

    def decoded() -> Iterator[tuple[float, Image.Image]]:
        try:
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                readable, pixels = capture.read()
                if not readable or pixels is None:
                    raise MediaError("Some sampled video frames could not be decoded; the file may be damaged.")
                _check_dimensions(pixels.shape[1], pixels.shape[0])
                frame = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
                yield round(index / fps, 4), frame
        except cv2.error as exc:
            raise MediaError("The video could not be safely decoded.") from exc
        finally:
            capture.release()

    # Decode one frame at a time; never retain 12 full-resolution RGB frames.
    return metadata, decoded()


def _quality(image: Image.Image) -> dict[str, float]:
    preview = image.copy()
    preview.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    gray = cv2.cvtColor(np.asarray(preview), cv2.COLOR_RGB2GRAY)
    return {
        "brightness": round(float(gray.mean()), 2),
        "contrast": round(float(gray.std()), 2),
        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
    }


def _faces(image: Image.Image) -> list[dict[str, int]]:
    """Return approximate face boxes in original-image coordinates."""
    preview = image.copy()
    preview.thumbnail((640, 640), Image.Resampling.LANCZOS)
    gray = cv2.cvtColor(np.asarray(preview), cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    )
    if cascade.empty():
        raise MediaError("The local face detector is unavailable; reinstall the OpenCV dependency.")
    boxes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    scale_x, scale_y = image.width / preview.width, image.height / preview.height
    return [
        {"x": round(x * scale_x), "y": round(y * scale_y),
         "width": round(width * scale_x), "height": round(height * scale_y)}
        for x, y, width, height in boxes[:100]
    ]


def _prediction(raw: dict[str, Any], classes: tuple[str, ...] = CLASSES) -> dict[str, Any]:
    try:
        if (not classes or len(set(classes)) != len(classes)
                or not set(classes).issubset(CLASSES)
                or set(raw["probabilities"]) != set(classes)):
            raise ValueError("Unsupported model classes")
        probabilities = {key: float(raw["probabilities"][key]) for key in classes}
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in probabilities.values()):
            raise ValueError("Invalid probabilities")
        if not math.isclose(sum(probabilities.values()), 1, abs_tol=0.005):
            raise ValueError("Probabilities do not sum to one")
        label = max(probabilities, key=probabilities.get)
        return {"label": label, "confidence": probabilities[label], "probabilities": probabilities}
    except (KeyError, TypeError, ValueError) as exc:
        raise InferenceError("The configured model returned invalid class probabilities.") from exc


def _save_preview(image: Image.Image, destination: Path) -> None:
    preview = image.convert("RGB")
    preview.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    preview.save(destination, format="JPEG", quality=88)


def analyze_media(
    path: Path, directory: Path, analysis_id: str, filename: str,
    size_bytes: int, detector: Any,
) -> dict[str, Any]:
    extension = Path(filename).suffix.lower()
    if extension in IMAGE_EXTENSIONS:
        metadata, decoded = _image_frames(path)
        media_type = "image"
    elif extension in VIDEO_EXTENSIONS:
        metadata, decoded = _video_frames(path, extension)
        media_type = "video"
    else:
        raise MediaError("Upload a JPEG, PNG, WebP, MP4, MOV, AVI, or WebM file.")
    metadata["size_bytes"] = size_bytes
    classes = tuple(detector.info.get("classes", CLASSES))
    frames: list[dict[str, Any]] = []
    try:
        for index, (timestamp, image) in enumerate(decoded):
            try:
                frame = {
                    "index": index, "timestamp_seconds": timestamp,
                    "thumbnail_url": f"/api/analyses/{analysis_id}/frames/{index}",
                    "heatmap_url": None, "prediction": None, "quality": _quality(image),
                    "faces": _faces(image),
                }
                frame["face_count"] = len(frame["faces"])
                _save_preview(image, directory / f"frame-{index}.jpg")
                if detector.ready:
                    try:
                        raw_prediction, heatmap = detector.predict_with_heatmap(image)
                        frame["prediction"] = _prediction(raw_prediction, classes)
                        if heatmap is not None:
                            _save_preview(heatmap, directory / f"heatmap-{index}.jpg")
                            heatmap.close()
                            frame["heatmap_url"] = f"/api/analyses/{analysis_id}/heatmaps/{index}"
                    except InferenceError:
                        raise
                    except Exception as exc:
                        raise InferenceError("The configured model could not analyze this media.") from exc
                frames.append(frame)
            finally:
                image.close()
    finally:
        if hasattr(decoded, "close"):
            decoded.close()
    prediction = None
    if detector.ready:
        means = {key: sum(frame["prediction"]["probabilities"][key] for frame in frames) / len(frames) for key in classes}
        prediction = _prediction({"probabilities": means}, classes)
    summary = {"sampled_frames": len(frames), "faces_detected": sum(frame["face_count"] for frame in frames),
               "frames_with_faces": sum(bool(frame["faces"]) for frame in frames)}
    for key in ("brightness", "contrast", "sharpness"):
        summary[f"average_{key}"] = round(sum(frame["quality"][key] for frame in frames) / len(frames), 2)
    limitations = [
        "Brightness, contrast, and sharpness describe image quality; they are not evidence of AI generation or manipulation.",
        "Media classification is probabilistic and cannot establish authenticity or provenance on its own.",
        "Face boxes use a classical frontal-face detector and can miss faces or include false detections. The baseline classifier analyzes the whole frame, not cropped faces.",
    ]
    if any(frame["heatmap_url"] for frame in frames):
        limitations.append("Grad-CAM heatmaps show model attention, not proof or boundaries of tampering.")
    if not detector.ready:
        limitations.append("No trained model is configured. This report contains media metadata and quality measurements only; no authenticity prediction was made.")
    limitations.extend(detector.info.get("limitations", []))
    if media_type == "video":
        limitations.append("Video analysis samples up to 12 evenly spaced frames and averages their class probabilities. Unsampled events and audio are not analyzed.")
    return {
        "id": analysis_id, "created_at": datetime.now(timezone.utc).isoformat(),
        "filename": filename, "media_type": media_type,
        "status": "complete" if detector.ready else "model_required",
        "prediction": prediction, "model": {**detector.info, "ready": bool(detector.ready)},
        "media": metadata, "frames": frames, "summary": summary, "limitations": limitations,
    }

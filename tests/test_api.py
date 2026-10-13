"""Integration checks for actual media, storage, and bounded upload behavior."""

from __future__ import annotations

import io
import struct
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from truthlens.app import create_app
from truthlens.media import CLASSES


class UntrainedDetector:
    ready = False
    info = {"ready": False, "status": "unconfigured", "classes": list(CLASSES)}


class TrainedDetector:
    ready = True
    info = {"ready": True, "status": "ready", "classes": list(CLASSES), "architecture": "test-double"}

    def predict_with_heatmap(self, image):
        probability = float(np.asarray(image).mean()) / 255
        values = {"authentic": 0.1, "ai_generated": probability * 0.8,
                  "manipulated": 0.9 - probability * 0.8}
        label = max(values, key=values.get)
        return (
            {"label": label, "probabilities": values, "confidence": values[label]},
            Image.new("RGB", image.size, "red"),
        )


def png_bytes(color=(120, 120, 120), size=(80, 64)) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, "PNG")
    return stream.getvalue()


def create_video(path: Path, frame_count=15, fps=5) -> bytes:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (80, 64))
    assert writer.isOpened(), "OpenCV must provide AVI/MJPEG decoding for the documented video workflow"
    try:
        for index in range(frame_count):
            writer.write(np.full((64, 80, 3), index * 12, dtype=np.uint8))
    finally:
        writer.release()
    return path.read_bytes()


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "data"


@pytest.fixture
def client(data_dir):
    with TestClient(create_app(data_dir=data_dir, detector=UntrainedDetector())) as test_client:
        yield test_client


def test_status_separates_service_readiness_from_model_readiness(client):
    response = client.get("/api/status")
    assert response.status_code == 200
    status = response.json()
    assert status["ready"] is True
    assert status["model"]["ready"] is False
    assert status["model"]["classes"] == list(CLASSES)
    assert status["limits"]["max_upload_bytes"] == 50 * 1024 * 1024
    assert ".webm" in status["supported"]["videos"]


def test_real_image_metadata_evidence_and_report_persist(client, data_dir):
    source = png_bytes()
    response = client.post("/api/analyze", files={"file": ("sample.png", source, "image/png")})
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["status"] == "model_required"
    assert result["prediction"] is None
    assert result["media"] == {
        "format": "PNG", "width": 80, "height": 64,
        "duration_seconds": None, "frame_count": 1, "fps": None,
        "size_bytes": len(source),
    }
    assert result["summary"]["average_brightness"] == 120
    assert result["summary"]["average_sharpness"] == 0
    assert result["summary"]["faces_detected"] == 0
    assert result["frames"][0]["faces"] == []
    assert result["frames"][0]["heatmap_url"] is None
    assert result["frames"][0]["prediction"] is None
    evidence = client.get(result["frames"][0]["thumbnail_url"])
    assert evidence.status_code == 200
    assert Image.open(io.BytesIO(evidence.content)).size == (80, 64)
    assert client.get(f'/api/analyses/{result["id"]}').json() == result
    report = client.get(f'/api/analyses/{result["id"]}/report')
    assert report.status_code == 200
    assert report.json() == result
    assert "attachment" in report.headers["content-disposition"]
    assert str(data_dir) not in report.text
    history = client.get("/api/analyses").json()
    assert history["total"] == 1
    assert history["analyses"][0]["id"] == result["id"]
    assert "frames" not in history["analyses"][0]
    assert not list(data_dir.rglob("upload*")), "Uploaded originals must be removed after processing"
    with TestClient(create_app(data_dir=data_dir, detector=UntrainedDetector())) as restarted:
        assert restarted.get(f'/api/analyses/{result["id"]}').json() == result


def test_filename_is_sanitized_and_paths_cannot_escape(client, data_dir):
    response = client.post("/api/analyze", files={"file": ("../../outside.png", png_bytes(), "image/png")})
    assert response.status_code == 201
    assert response.json()["filename"] == "outside.png"
    assert not (data_dir.parent / "outside.png").exists()
    assert client.get("/api/analyses/not-a-uuid").status_code == 422
    analysis_id = response.json()["id"]
    assert client.get(f"/api/analyses/{analysis_id}/frames/-1").status_code == 404
    assert client.get(f"/api/analyses/{analysis_id}/frames/1").status_code == 404
    assert client.get(f"/api/analyses/{analysis_id}/heatmaps/0").status_code == 404


@pytest.mark.parametrize(
    "filename, payload, status",
    [("sample.txt", b"hello", 400), ("sample.png", b"not an image", 400),
     ("sample.png", b"", 400), ("sample.mp4", png_bytes(), 400),
     ("sample.avi", b"invalid video", 400)],
)
def test_invalid_uploads_never_create_history_or_leave_files(client, data_dir, filename, payload, status):
    response = client.post("/api/analyze", files={"file": (filename, payload)})
    assert response.status_code == status
    assert "detail" in response.json()
    assert client.get("/api/analyses").json()["total"] == 0
    assert not list((data_dir / "results").glob("*"))


def test_unsupported_image_format_disguised_as_png_is_rejected(client):
    buffer = io.BytesIO()
    Image.new("RGB", (20, 20)).save(buffer, "GIF")
    response = client.post("/api/analyze", files={"file": ("sample.png", buffer.getvalue())})
    assert response.status_code == 400
    assert "JPEG" in response.json()["detail"]


def test_oversized_dimensions_are_rejected_before_pixel_decode(client):
    header = struct.pack(">IIBBBBB", 6000, 6000, 8, 2, 0, 0, 0)
    chunk = b"IHDR" + header
    source = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", len(header)) + chunk + struct.pack(">I", zlib.crc32(chunk))
    empty_data = b"IDAT" + zlib.compress(b"")
    source += struct.pack(">I", len(empty_data) - 4) + empty_data + struct.pack(">I", zlib.crc32(empty_data))
    response = client.post("/api/analyze", files={"file": ("large.png", source)})
    assert response.status_code == 400
    assert "25,000,000" in response.json()["detail"]


def test_animated_image_is_rejected(client):
    stream = io.BytesIO()
    image = Image.new("RGB", (16, 16), "red")
    image.save(stream, "PNG", save_all=True, append_images=[Image.new("RGB", (16, 16), "blue")], duration=100)
    response = client.post("/api/analyze", files={"file": ("animation.png", stream.getvalue())})
    assert response.status_code == 400
    assert "Animated" in response.json()["detail"]


def test_image_upload_byte_limit_is_enforced_and_cleans_up(data_dir):
    source = io.BytesIO()
    pixels = np.random.default_rng(1).integers(0, 256, (80, 80, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(source, "PNG")
    with TestClient(create_app(data_dir=data_dir, detector=UntrainedDetector(), max_upload_bytes=1024)) as client:
        response = client.post("/api/analyze", files={"file": ("large.png", source.getvalue())})
        assert response.status_code == 413
        assert client.get("/api/analyses").json()["total"] == 0
    assert not list((data_dir / "results").glob("*"))


def test_request_limit_handles_unknown_content_length(data_dir):
    body = b'--test\r\nContent-Disposition: form-data; name="file"; filename="large.png"\r\nContent-Type: image/png\r\n\r\n' + b"x" * 70000 + b"\r\n--test--\r\n"
    with TestClient(create_app(data_dir=data_dir, detector=UntrainedDetector(), max_upload_bytes=1024)) as client:
        response = client.post(
            "/api/analyze", content=iter([body[:30000], body[30000:]]),
            headers={"Content-Type": "multipart/form-data; boundary=test"},
        )
        assert response.status_code == 413
        assert client.get("/api/analyses").json()["total"] == 0


def test_real_video_sampling_returns_timestamps_and_metadata(client, tmp_path):
    source = create_video(tmp_path / "clip.avi")
    response = client.post("/api/analyze", files={"file": ("clip.avi", source, "video/x-msvideo")})
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["media_type"] == "video"
    assert result["media"]["width"] == 80
    assert result["media"]["height"] == 64
    assert result["media"]["frame_count"] == 15
    assert result["media"]["duration_seconds"] == 3
    assert result["summary"]["sampled_frames"] == 12
    assert result["frames"][0]["timestamp_seconds"] == 0
    assert result["frames"][-1]["timestamp_seconds"] == 2.8
    assert result["frames"][0]["quality"]["brightness"] < result["frames"][-1]["quality"]["brightness"]
    for frame in result["frames"]:
        assert client.get(frame["thumbnail_url"]).status_code == 200


@pytest.mark.parametrize(
    "width,height,fps,frame_count,expected",
    [(9000, 64, 5, 15, "pixels"), (80, 64, 5, 3001, "10 minutes"),
     (80, 64, 0, 15, "frame rate")],
)
def test_unsafe_video_metadata_is_rejected_before_frame_decode(
    client, monkeypatch, width, height, fps, frame_count, expected,
):
    class UnsafeCapture:
        released = False

        def isOpened(self):
            return True

        def get(self, key):
            return {cv2.CAP_PROP_FRAME_WIDTH: width, cv2.CAP_PROP_FRAME_HEIGHT: height,
                    cv2.CAP_PROP_FPS: fps, cv2.CAP_PROP_FRAME_COUNT: frame_count}[key]

        def read(self):
            raise AssertionError("Unsafe metadata must be rejected before decoding any frame")

        def release(self):
            self.released = True

    capture = UnsafeCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: capture)
    response = client.post("/api/analyze", files={"file": ("unsafe.avi", b"RIFF\x00\x00\x00\x00AVI ")})
    assert response.status_code == 400
    assert expected in response.json()["detail"]
    assert capture.released


def test_trained_model_prediction_heatmap_and_video_aggregation(data_dir, tmp_path):
    with TestClient(create_app(data_dir=data_dir, detector=TrainedDetector())) as client:
        image = client.post("/api/analyze", files={"file": ("white.png", png_bytes((255, 255, 255)))})
        assert image.status_code == 201, image.text
        result = image.json()
        assert result["status"] == "complete"
        assert result["prediction"]["label"] == "ai_generated"
        assert result["prediction"]["confidence"] == pytest.approx(0.8)
        heatmap = client.get(result["frames"][0]["heatmap_url"])
        assert heatmap.status_code == 200
        assert Image.open(io.BytesIO(heatmap.content)).size == (80, 64)
        video = client.post("/api/analyze", files={"file": ("clip.avi", create_video(tmp_path / "clip.avi"))})
        assert video.status_code == 201, video.text
        analysis = video.json()
        assert set(analysis["prediction"]["probabilities"]) == set(CLASSES)
        for key in CLASSES:
            expected = np.mean([frame["prediction"]["probabilities"][key] for frame in analysis["frames"]])
            assert analysis["prediction"]["probabilities"][key] == pytest.approx(expected)


def test_model_inference_failure_is_safe_and_leaves_no_partial_record(data_dir):
    class FailingDetector(TrainedDetector):
        def predict_with_heatmap(self, image):
            raise RuntimeError("private details /tmp/secret-model")

    with TestClient(create_app(data_dir=data_dir, detector=FailingDetector())) as client:
        response = client.post("/api/analyze", files={"file": ("image.png", png_bytes())})
        assert response.status_code == 503
        assert "secret-model" not in response.text
        assert client.get("/api/analyses").json()["total"] == 0
    assert not list((data_dir / "results").glob("*"))


def test_binary_pretrained_scores_reports_and_video_without_heatmaps(data_dir, tmp_path):
    class BinaryDetector:
        ready = True
        info = {"ready": True, "classes": ["authentic", "ai_generated"],
                "model_id": "test/binary", "capabilities": ["ai_generation"],
                "limitations": ["Does not detect face swaps or general manipulation."]}

        def predict_with_heatmap(self, image):
            score = float(np.asarray(image).mean()) / 255
            probabilities = {"authentic": 1 - score, "ai_generated": score}
            label = max(probabilities, key=probabilities.get)
            return {"label": label, "confidence": probabilities[label], "probabilities": probabilities}, None

    with TestClient(create_app(data_dir=data_dir, detector=BinaryDetector())) as client:
        for filename, payload in (
            ("white.png", png_bytes((255, 255, 255))),
            ("clip.avi", create_video(tmp_path / "binary.avi")),
        ):
            response = client.post("/api/analyze", files={"file": (filename, payload)})
            assert response.status_code == 201, response.text
            result = response.json()
            assert set(result["prediction"]["probabilities"]) == {"authentic", "ai_generated"}
            assert all(frame["heatmap_url"] is None for frame in result["frames"])
            assert any("face swaps" in text for text in result["limitations"])
            assert not any("Grad-CAM" in text for text in result["limitations"])
            for label in ("authentic", "ai_generated"):
                expected = np.mean([frame["prediction"]["probabilities"][label] for frame in result["frames"]])
                assert result["prediction"]["probabilities"][label] == pytest.approx(expected)
            assert client.get(f'/api/analyses/{result["id"]}/report').json() == result
            assert client.get(f'/api/analyses/{result["id"]}/heatmaps/0').status_code == 404


def test_app_selects_pretrained_bundle_and_explicit_checkpoint_has_priority(data_dir, tmp_path, monkeypatch):
    import truthlens.pretrained as pretrained_module

    directory = tmp_path / "pretrained"
    directory.mkdir()
    (directory / "manifest.json").write_text("{}")
    monkeypatch.setenv("TRUTHLENS_PRETRAINED_DIR", str(directory))
    loaded = []

    def load_bundle(path):
        loaded.append(path)
        return UntrainedDetector()

    monkeypatch.setattr(pretrained_module, "PretrainedDetector", load_bundle)
    with TestClient(create_app(data_dir=data_dir)) as client:
        assert client.get("/api/status").status_code == 200
    assert loaded == [directory]
    with TestClient(create_app(data_dir=data_dir, model_path=tmp_path / "missing.pt")) as client:
        assert client.get("/api/status").json()["model"]["status"] == "load_failed"
    assert loaded == [directory], "An explicit custom checkpoint must take priority over the pretrained bundle"


def test_missing_checkpoint_keeps_metadata_workflow_available(data_dir, tmp_path):
    with TestClient(create_app(data_dir=data_dir, model_path=tmp_path / "missing.pt")) as client:
        status = client.get("/api/status").json()
        assert status["ready"] is True
        assert status["model"]["status"] == "load_failed"
        assert str(tmp_path) not in str(status)
        response = client.post("/api/analyze", files={"file": ("image.png", png_bytes())})
        assert response.status_code == 201
        assert response.json()["prediction"] is None


def test_concurrency_has_bounded_active_analyses(data_dir, monkeypatch):
    import truthlens.app as app_module

    original = app_module.analyze_media
    started = threading.Event()
    release = threading.Event()
    count_lock = threading.Lock()
    active = 0

    def blocking_analysis(*args):
        nonlocal active
        with count_lock:
            active += 1
            if active == 2:
                started.set()
        assert release.wait(timeout=10)
        return original(*args)

    monkeypatch.setattr(app_module, "analyze_media", blocking_analysis)
    with TestClient(create_app(data_dir=data_dir, detector=UntrainedDetector())) as client:
        with ThreadPoolExecutor(max_workers=2) as executor:
            pending = [executor.submit(client.post, "/api/analyze", files={"file": ("image.png", png_bytes())}) for _ in range(2)]
            try:
                assert started.wait(timeout=5)
                rejected = client.post("/api/analyze", files={"file": ("image.png", png_bytes())})
                assert rejected.status_code == 429
                assert client.get("/api/status").status_code == 200
            finally:
                release.set()
            assert all(future.result().status_code == 201 for future in pending)

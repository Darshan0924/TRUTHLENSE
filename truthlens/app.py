"""FastAPI service and static dashboard for TruthLens."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from truthlens.media import (
    CLASSES, IMAGE_EXTENSIONS, MAX_DIMENSION, MAX_FRAMES, MAX_PIXELS,
    MAX_UPLOAD_BYTES, MAX_VIDEO_SECONDS, VIDEO_EXTENSIONS,
    InferenceError, MediaError, analyze_media,
)
from truthlens.storage import AnalysisStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class _UploadTooLarge(Exception):
    pass


class UploadLimitMiddleware:
    """Cap multipart bytes before Starlette can spool an unbounded upload."""

    def __init__(self, app: Any, max_bytes: int):
        self.app = app
        # A small allowance covers multipart headers, never unlimited extra fields.
        self.max_request_bytes = max_bytes + 64 * 1024

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST" or scope.get("path") != "/api/analyze":
            await self.app(scope, receive, send)
            return
        error = JSONResponse({"detail": "The upload exceeds the allowed size."}, status_code=413)
        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    if int(value) > self.max_request_bytes:
                        await error(scope, receive, send)
                        return
                except ValueError:
                    pass
        seen = 0
        exceeded = False
        error_sent = False

        async def limited_receive() -> Any:
            nonlocal seen, exceeded
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_request_bytes:
                    exceeded = True
                    raise _UploadTooLarge()
            return message

        async def limited_send(message: Any) -> None:
            nonlocal error_sent
            if exceeded:
                if not error_sent:
                    error_sent = True
                    await error(scope, receive, send)
                return
            await send(message)

        try:
            await self.app(scope, limited_receive, limited_send)
        except _UploadTooLarge:
            if not error_sent:
                await error(scope, receive, send)


class _UnavailableDetector:
    ready = False

    def __init__(self, error: str):
        self.info = {
            "ready": False, "status": "load_failed", "classes": list(CLASSES),
            "error": error,
        }


def _filename(value: str | None) -> str:
    name = (value or "upload").replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(character for character in name if character.isprintable())
    if not name or len(name) > 180:
        raise HTTPException(400, "The upload filename must contain 1 to 180 printable characters.")
    return name


def create_app(
    data_dir: Path | None = None, model_path: Path | None = None,
    detector: Any = None, max_upload_bytes: int = MAX_UPLOAD_BYTES,
) -> FastAPI:
    """Factory with injectable detector and isolated storage for tests."""
    configured_data = data_dir or Path(os.environ.get("TRUTHLENS_DATA_DIR", str(PROJECT_ROOT / ".truthlens")))
    configured_model = model_path
    if configured_model is None and os.environ.get("TRUTHLENS_MODEL_PATH"):
        configured_model = Path(os.environ["TRUTHLENS_MODEL_PATH"])

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.store = await run_in_threadpool(AnalysisStore, configured_data)
        if detector is not None:
            application.state.detector = detector
        else:
            try:
                from truthlens.ml import Detector

                application.state.detector = await run_in_threadpool(Detector, configured_model)
            except Exception:
                # Low-level checkpoint errors can contain local paths. Keep the
                # public status actionable without echoing filesystem details.
                application.state.detector = _UnavailableDetector(
                    "The configured checkpoint could not be loaded. Check its format and file permissions, then restart."
                )
        application.state.analysis_slots = asyncio.Semaphore(2)
        # Bound CPU and memory use while producing model predictions and Grad-CAM.
        application.state.model_lock = asyncio.Lock()
        yield

    application = FastAPI(title="TruthLens", version="0.1.0", lifespan=lifespan)
    application.add_middleware(UploadLimitMiddleware, max_bytes=max_upload_bytes)

    @application.get("/api/status")
    async def status() -> dict[str, Any]:
        current = application.state.detector
        return {
            "ready": True, "model": {**current.info, "ready": bool(current.ready)},
            "limits": {"max_upload_bytes": max_upload_bytes, "max_video_seconds": MAX_VIDEO_SECONDS,
                       "max_frames": MAX_FRAMES, "max_pixels": MAX_PIXELS, "max_dimension": MAX_DIMENSION},
            "supported": {"images": sorted(IMAGE_EXTENSIONS), "videos": sorted(VIDEO_EXTENSIONS)},
        }

    @application.post("/api/analyze", status_code=201)
    async def analyze(file: UploadFile = File(...)) -> dict[str, Any]:
        slots = application.state.analysis_slots
        acquired = False
        directory = None
        succeeded = False
        try:
            filename = _filename(file.filename)
            extension = Path(filename).suffix.lower()
            if extension not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
                raise HTTPException(400, "Upload a JPEG, PNG, WebP, MP4, MOV, AVI, or WebM file.")
            try:
                await asyncio.wait_for(slots.acquire(), timeout=0.02)
                acquired = True
            except TimeoutError as exc:
                raise HTTPException(429, "Two analyses are already running. Try again shortly.") from exc
            analysis_id = str(uuid4())
            store = application.state.store
            directory = store.directory(analysis_id)
            await run_in_threadpool(directory.mkdir, parents=True, exist_ok=False)
            upload_path = directory / f"upload{extension}"
            size = 0
            with upload_path.open("wb") as destination:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_upload_bytes:
                        raise HTTPException(413, "The upload exceeds the allowed size.")
                    await run_in_threadpool(destination.write, chunk)
            if size == 0:
                raise HTTPException(400, "The uploaded file is empty.")
            current = application.state.detector
            if current.ready:
                async with application.state.model_lock:
                    result = await run_in_threadpool(
                        analyze_media, upload_path, directory, analysis_id, filename, size, current,
                    )
            else:
                result = await run_in_threadpool(
                    analyze_media, upload_path, directory, analysis_id, filename, size, current,
                )
            await run_in_threadpool(upload_path.unlink)
            await run_in_threadpool(store.save, result)
            succeeded = True
            return result
        except MediaError as exc:
            raise HTTPException(400, str(exc)) from exc
        except InferenceError as exc:
            raise HTTPException(503, str(exc)) from exc
        finally:
            await file.close()
            if directory is not None and not succeeded:
                await run_in_threadpool(shutil.rmtree, directory, ignore_errors=True)
            if acquired:
                slots.release()

    @application.get("/api/analyses")
    async def recent(limit: int = Query(20, ge=1, le=100)) -> dict[str, Any]:
        return await run_in_threadpool(application.state.store.recent, limit)

    async def find(analysis_id: UUID) -> dict[str, Any]:
        result = await run_in_threadpool(application.state.store.get, str(analysis_id))
        if result is None:
            raise HTTPException(404, "Analysis not found.")
        return result

    @application.get("/api/analyses/{analysis_id}")
    async def detail(analysis_id: UUID) -> dict[str, Any]:
        return await find(analysis_id)

    @application.get("/api/analyses/{analysis_id}/report")
    async def report(analysis_id: UUID) -> Response:
        result = await find(analysis_id)
        return Response(
            content=json.dumps(result, indent=2, allow_nan=False), media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="truthlens-{analysis_id}.json"'},
        )

    async def evidence(analysis_id: UUID, index: int, heatmap: bool) -> FileResponse:
        result = await find(analysis_id)
        if index < 0 or index >= len(result["frames"]):
            raise HTTPException(404, "Evidence frame not found.")
        if heatmap and not result["frames"][index]["heatmap_url"]:
            raise HTTPException(404, "No model heatmap is available for this frame.")
        prefix = "heatmap" if heatmap else "frame"
        path = application.state.store.directory(str(analysis_id)) / f"{prefix}-{index}.jpg"
        if not path.is_file():
            raise HTTPException(404, "The saved evidence frame is unavailable.")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    @application.get("/api/analyses/{analysis_id}/frames/{index}")
    async def frame(analysis_id: UUID, index: int) -> FileResponse:
        return await evidence(analysis_id, index, False)

    @application.get("/api/analyses/{analysis_id}/heatmaps/{index}")
    async def heatmap(analysis_id: UUID, index: int) -> FileResponse:
        return await evidence(analysis_id, index, True)

    @application.get("/")
    async def dashboard() -> FileResponse:
        path = PROJECT_ROOT / "static" / "index.html"
        if not path.is_file():
            raise HTTPException(503, "The dashboard files have not been installed.")
        return FileResponse(path)

    application.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static", check_dir=False), name="static")
    return application


app = create_app()

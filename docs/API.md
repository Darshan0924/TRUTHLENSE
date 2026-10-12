# API

Start the service from the repository root. FastAPI's generated API documentation is available at `/docs` on that service.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/status` | Service/model readiness, accepted extensions, processing limits |
| POST | `/api/analyze` | Multipart upload with one `file` field; returns an analysis (HTTP 201) |
| GET | `/api/analyses?limit=100` | Most recent summaries and total saved count |
| GET | `/api/analyses/{id}` | Full saved analysis |
| GET | `/api/analyses/{id}/report` | Download the analysis as JSON |
| GET | `/api/analyses/{id}/frames/{index}` | Saved JPEG preview of a sampled frame |
| GET | `/api/analyses/{id}/heatmaps/{index}` | Optional Grad-CAM attention visualization |

Analysis identifiers are UUIDs. Frame indices start at zero. Frame URLs are included in the analysis response.

Supported images: JPEG, PNG, and non-animated WebP. Supported video containers: MP4, MOV, AVI, and WebM; the codec must be decodable by OpenCV. The service caps uploads at 50 MiB, duration at 10 minutes, dimensions at 8,192 pixels per side and 25 million pixels total, and frame sampling at 12 frames. It supports two active media analyses; model inference is serialized to bound CPU/memory use. Original uploads are deleted after processing; preview frames and JSON records remain in local storage.

An unconfigured detector returns `status: "model_required"`, `prediction: null`, media metadata, sampled frames, actual quality measurements, and approximate frontal-face boxes. A configured model returns `status: "complete"`, a three-class probability mapping, the winning label, confidence, per-frame predictions, and attention maps. Video class probabilities are the arithmetic mean across sampled frames. These are model scores, not calibrated probabilities of authenticity.

`/api/status` distinguishes service readiness (`ready`) from classifier availability (`model.ready`). A missing or incompatible configured checkpoint makes the model unavailable while media inspection remains usable. Restart the service after changing the model path or checkpoint.

Errors use a JSON `detail` message: HTTP 400 for unsupported/corrupt media, 413 for size limits, 429 when analysis slots are full, 503 for a configured model failing inference, and 404 for missing reports/evidence. Invalid identifiers and query values return 422. Failed analyses do not enter history, and their temporary files are removed.

This local service has no authentication or tenant isolation. Every client that can reach it can view saved media previews and reports. Keep it bound to loopback until deployment controls are added.

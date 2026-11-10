import io
import os
import time
from collections import defaultdict, deque

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from app.inference import Classifier

MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "8")) * 1024 * 1024
ALLOWED = {"image/jpeg", "image/png", "image/webp"}
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "30"))
Image.MAX_IMAGE_PIXELS = 40_000_000  # reject decompression bombs

app = FastAPI(title="LeafLens API", version="1.0.0", docs_url="/api/docs", openapi_url="/api/openapi.json")
origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])

_classifier: Classifier | None = None
_hits: dict[str, deque] = defaultdict(deque)


def classifier() -> Classifier:
    global _classifier
    if _classifier is None:
        _classifier = Classifier()
    return _classifier


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path == "/api/predict" and RATE_LIMIT > 0:
        ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "unknown").split(",")[0].strip()
        now, q = time.monotonic(), _hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            return JSONResponse({"detail": "Too many requests. Please wait a minute and try again."}, status_code=429)
        q.append(now)
    return await call_next(request)


@app.get("/api/health")
def health():
    c = classifier()
    return {"status": "ok", "model": c.card["name"], "classes": len(c.labels), "confidence_threshold": c.threshold}


@app.get("/api/model")
def model_info():
    """Model card: what it was trained on and how well it does, including the honest field-photo number."""
    return classifier().card


@app.get("/api/classes")
def classes():
    c = classifier()
    return [{**c._meta(i)} for i in range(len(c.labels))]


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), top_k: int = Query(3, ge=1, le=5), explain: bool = Query(True)):
    if file.content_type not in ALLOWED:
        raise HTTPException(415, "Please upload a JPEG, PNG or WebP photo.")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"Photo is too large (limit {MAX_BYTES // (1024 * 1024)} MB).")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, "That file could not be read as an image.")
    if min(img.size) < 64:
        raise HTTPException(400, "Photo is too small; use at least 64 x 64 pixels.")
    return classifier().predict(img, top_k=top_k, explain=explain)

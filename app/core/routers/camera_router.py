"""
Маршрутизатор для трансляции видео с камеры и детекции лиц.
Поддерживает:
- Jetson + WebRTC (приоритет)
- OpenCV + MJPEG (fallback)
- Эндпоинт /detect — детекция лиц с возвратом JSON
"""
import glob
import logging
import os
import threading
from datetime import datetime
from typing import Optional, List

import cv2
import faiss
import numpy as np
from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from app.config.config import settings

# === Импорты для face_analysis ===
from app.core.face_analysis.core import get_pipeline

# Настройка логгера
logger = logging.getLogger("MBB_logger")

# Глобальные переменные
USE_JETSON_CAMERA = False
output_stream: Optional['jetson_utils.videoOutput'] = None
camera_source: Optional['jetson_utils.videoSource'] = None
stream_thread: Optional[threading.Thread] = None

# Блокировка и VideoCapture для OpenCV
lock = threading.Lock()
cap = None
camera_initialized = False
face_app = None  # Модель face_analysis
faiss_index = None

# --- Попытка импорта jetson_utils ---
try:
    import jetson_utils
    USE_JETSON_CAMERA = True
except ImportError:
    logger.warning("Module jetson_utils not found. Using OpenCV fallback.")

# --- Ленивая инициализация шаблонов ---
_templates: Optional[Jinja2Templates] = None


# Загрузка базы эмбеддингов
db_names_local = []
db_embeddings_local = []

def update_embeddings():
    global db_names_local
    global db_embeddings_local
    global cap, face_app, faiss_index

    logger.info("=====LOADING EMBEDDING DATABASE====")
    for file in sorted(glob.glob(f"{settings.EMBEDDINGS_DIR}/*.npy")):
        name = os.path.basename(file).replace(".npy", "")
        logger.info(f"=====>{name}")
        embedding = np.load(file)
        db_names_local.append(name)
        db_embeddings_local.append(embedding)
    logger.info("=====COMPLETE====")

    # Инициализация модели распознавания лиц
    try:
        face_app = get_pipeline()
        logger.info("Face analysis pipeline initialized")
    except Exception as e:
        logger.error(f"Failed to initialize face analysis model: {e}")

    # Создание FAISS индекса
    if db_embeddings_local:
        dim = db_embeddings_local[0].shape[0]
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(np.array(db_embeddings_local).astype('float32'))
    else:
        logger.warning("No embeddings found in 'embeddings/' directory.")
        faiss_index = None

def get_templates() -> Jinja2Templates:
    """
    Ленивая инициализация Jinja2Templates.
    Очищает кэш, чтобы избежать ошибок вроде 'unhashable type: dict'.
    """
    global _templates
    if _templates is None:
        doc_root = settings.CAMERA_DOC_ROOT
        if not doc_root:
            raise RuntimeError("CAMERA_DOC_ROOT is not set in settings")
        if not os.path.exists(doc_root):
            raise RuntimeError(f"Template directory does not exist: {doc_root}")

        _templates = Jinja2Templates(directory=doc_root)

        if hasattr(_templates.env, "cache") and _templates.env.cache is not None:
            _templates.env.cache.clear()
            logger.debug("Jinja2 template cache cleared to prevent 'unhashable type' error")

        logger.info(f"Templates initialized: {doc_root}")
    return _templates


# === Pydantic модели для ответа ===
class FaceDetection(BaseModel):
    bbox: List[int]  # [x1, y1, x2, y2]
    name: str
    confidence: float


class DetectionResponse(BaseModel):
    timestamp: str
    faces: List[FaceDetection]
    success: bool


# --- Инициализация роутера ---
router = APIRouter(prefix="/camera", tags=["camera"])


def init_camera() -> None:
    if USE_JETSON_CAMERA:
        return
    """Ленивая инициализация OpenCV камеры."""
    global cap, camera_initialized
    if not camera_initialized:
        with lock:
            if not camera_initialized:
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 15)
                if not cap.isOpened():
                    logger.error("Failed to open camera via OpenCV")
                else:
                    logger.info("OpenCV camera initialized")
                camera_initialized = True


def get_frame() -> Optional[bytes]:
    """Захватывает и кодирует один кадр с меткой времени."""
    global cap
    init_camera()

    ret, img = cap.read()
    if not ret or img is None:
        logger.warning("Failed to capture frame from camera")
        return None

    # Добавляем текущее время на кадр
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, str(datetime.now().time()), (10, 50), font, 1, (255, 255, 255), 2)

    # Кодируем в JPEG
    success, jpg = cv2.imencode('.jpg', img)
    return jpg.tobytes() if success else None


def generate_mjpeg_stream():
    """Генератор MJPEG-потока."""
    while True:
        frame = get_frame()
        if frame is None:
            continue
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame + b'\r\n'
        )
        threading.Event().wait(0.05)  # ~20 FPS


@router.on_event("startup")
async def startup_event() -> None:
    """Инициализация модели face_analysis и запуск потока (Jetson или OpenCV)."""
    global stream_thread

    update_embeddings()

    # Запуск потока для Jetson
    if USE_JETSON_CAMERA:
        stream_thread = threading.Thread(target=run_jetson_stream, daemon=True)
        stream_thread.start()
        logger.info("Jetson WebRTC stream started")
    else:
        logger.info("Fallback: using MJPEG via OpenCV")


def run_jetson_stream() -> None:
    """Фоновая трансляция через Jetson (WebRTC)."""
    global output_stream, camera_source
    while True:
        try:
            camera_source = jetson_utils.videoSource("csi://0")
            port = settings.CAMERA_PORT
            if not port:
                logger.error("CAMERA_PORT is not set")
                break
            output_stream = jetson_utils.videoOutput(f"webrtc://@:{port}/my_stream")
            logger.info(f"WebRTC streaming started: webrtc://@:{port}/my_stream")

            while True:
                img = camera_source.Capture()
                if img is None:
                    logger.warning("Frame not captured from Jetson camera")
                    continue
                output_stream.Render(img)
                fps = output_stream.GetFrameRate()
                logger.debug(f"Streaming @ {fps:.1f} FPS via WebRTC")

        except Exception as e:
            logger.error(f"Jetson stream error: {e}")
            if output_stream:
                try:
                    output_stream.Close()
                except Exception:
                    pass
            if camera_source:
                try:
                    camera_source.Close()
                except Exception:
                    pass
            threading.Event().wait(2)


# === ЭНДПОИНТЫ ===

@router.get("/", response_class=HTMLResponse)
async def camera_page(request: Request) -> HTMLResponse:
    """
    Основная страница с видеотрансляцией.
    Показывает ссылку на WebRTC или MJPEG-поток.
    """
    host = request.client.host
    port = settings.CAMERA_PORT
    if not port:
        logger.error("CAMERA_PORT is not set")
        return HTMLResponse(content="Server configuration error", status_code=500)

    if USE_JETSON_CAMERA:
        stream_url = f"http://{host}:{port}/my_stream"
        content = f"""
            <!DOCTYPE html>
            <html>
            <head><title>Camera</title></head>
            <body>
                <h1>Live Stream</h1>
                <p>WebRTC: <a href="{stream_url}">{stream_url}</a></p>
            </body>
            </html>
        """
    else:
        content = f"""
            <!DOCTYPE html>
            <html>
            <head><title>Camera</title></head>
            <body>
                <h1>Live Stream</h1>
                <img src="/camera/video" alt="MJPEG Stream" style="max-width:100%" />
            </body>
            </html>
        """
    return HTMLResponse(content=content, status_code=200)


@router.get("/video", response_class=Response)
async def video_feed() -> StreamingResponse:
    """
    MJPEG поток для браузера (работает без Jetson).
    Используется как src у <img>.
    """
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/detect", response_model=DetectionResponse)
async def detect_faces(response: Response):
    """
    Эндпоинт для детекции лиц.
    Использует jetson_utils при наличии, иначе OpenCV.
    Возвращает JSON с bbox, именем и уверенностью.
    """
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    global cap, camera_source, face_app, faiss_index

    if face_app is None:
        logger.error("Face analysis model not initialized")
        return DetectionResponse(timestamp=str(datetime.now()), faces=[], success=False)

    # === Захват кадра: jetson_utils или OpenCV ===
    frame = None

    if USE_JETSON_CAMERA:
        # Попробуем захватить через jetson_utils
        try:
            if camera_source is None:
                camera_source = jetson_utils.videoSource("csi://0")
            img = camera_source.Capture(timeout=1000)  # таймаут 1 сек
            if img is not None:
                # Преобразуем CUDA image → numpy array
                frame = jetson_utils.cudaToNumpy(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        except Exception as e:
            logger.warning(f"Failed to capture via jetson_utils: {e}")
    else:
        # Fallback: OpenCV
        init_camera()
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                frame = None

    if frame is None:
        logger.warning("Failed to capture frame from any source")
        return DetectionResponse(timestamp=str(datetime.now()), faces=[], success=False)

    # === Обработка кадра (одинаково для обоих источников) ===
    small_frame = cv2.resize(frame, (640, 480))
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Детекция лиц
    faces = face_app.get(rgb_frame)
    if not faces:
        return DetectionResponse(
            timestamp=str(datetime.now().isoformat()),
            faces=[],
            success=True
        )

    # Распознавание
    detections = []
    h_ratio = frame.shape[0] / 480
    w_ratio = frame.shape[1] / 640

    for face in faces:
        bbox = face.bbox.astype(int)
        query_emb = face.normed_embedding.reshape(1, -1).astype('float32')

        if faiss_index is not None and len(db_embeddings_local) > 0:
            distances, indices = faiss_index.search(query_emb, k=1)
            score = distances[0][0]
            idx = indices[0][0]

            if score > settings.CAMERA_DETECTION_THRESHOLD:
                name = db_names_local[idx]
            else:
                name = "Unknown"
            confidence = float(score)
        else:
            name = "Unknown"
            confidence = 0.0

        # Масштабирование bbox к оригинальному размеру
        bbox_orig = [
            int(bbox[0] * w_ratio),
            int(bbox[1] * h_ratio),
            int(bbox[2] * w_ratio),
            int(bbox[3] * h_ratio)
        ]

        detections.append(FaceDetection(
            bbox=bbox_orig,
            name=name,
            confidence=confidence
        ))

    return DetectionResponse(
        timestamp=str(datetime.now().isoformat()),
        faces=detections,
        success=True
    )
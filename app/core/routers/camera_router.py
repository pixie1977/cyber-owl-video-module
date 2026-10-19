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
import time
from datetime import datetime
from typing import Optional, List, Any

import cv2
import faiss
import numpy as np
from fastapi import APIRouter, Request, Response, HTTPException, status
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.config.config import settings
# === Импорты для face_analysis ===
from app.core.face_analysis.core import get_pipeline
from app.integration.integration_adapter import PeriodicFacesDataSender
from app.core.meeting_manager import MeetingManager

# === URL для LLM-модуля (должен быть определён ДО использования) ===
LLM_ENDPOINT_URL: str = f"http://{settings.LLM_MODULE_HOST}:{settings.LLM_MODULE_PORT}/image_detect"

# Настройка логгера
logger = logging.getLogger("MBB_logger")

# Глобальные переменные
USE_JETSON_CAMERA = False
output_stream: Optional['jetson_utils.videoOutput'] = None
camera_source: Optional['jetson_utils.videoSource'] = None
stream_thread: Optional[threading.Thread] = None

# Блокировка и VideoCapture для OpenCV
lock = threading.Lock()
face_lock = threading.RLock()  # Для защиты face_app, faiss_index, db_names_local
cam_lock = threading.Lock()  # Для защиты camera_source.Capture() из MJPEG и WebRTC
cap = None
cap_register = None  # Отдельная камера для регистрации
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
    global face_app, faiss_index

    logger.info("=====LOADING EMBEDDING DATABASE====")
    # Очищаем перед загрузкой, чтобы не дублировать
    db_names_local.clear()
    db_embeddings_local.clear()

    for file in sorted(glob.glob(f"{settings.EMBEDDINGS_DIR}/*.npy")):
        name = os.path.basename(file).replace(".npy", "")
        logger.info(f"=====>{name}")
        embedding = np.load(file)
        db_names_local.append(name)
        db_embeddings_local.append(embedding)
    logger.info(f"=====COMPLETE: {len(db_names_local)} embeddings loaded=====")

    # Инициализация модели распознавания лиц (ВНЕ face_lock — модель грузится долго)
    try:
        face_app = get_pipeline()
        logger.info("Face analysis pipeline initialized")
    except Exception as e:
        logger.error(f"Failed to initialize face analysis model: {e}")

    # Создание FAISS индекса (защищено face_lock — быстрая операция)
    with face_lock:
        if db_embeddings_local:
            dim = db_embeddings_local[0].shape[0]
            faiss_index = faiss.IndexFlatIP(dim)
            faiss_index.add(np.array(db_embeddings_local).astype('float32'))
            logger.info(f"FAISS index created with {len(db_embeddings_local)} vectors")
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
    age: int
    sex: str
    confidence: float

    def to_dict(self):
        return self.model_dump()


class DetectionResponse(BaseModel):
    timestamp: str
    faces: List[FaceDetection]
    success: bool


class RegisterFaceRequest(BaseModel):
    username: str


# --- Инициализация роутера ---
router = APIRouter(tags=["camera"])


def init_camera(register: bool = False) -> None:
    """Ленивая инициализация OpenCV камеры.
    
    :param register: если True, инициализирует отдельную камеру для регистрации
    """
    if USE_JETSON_CAMERA:
        return
    
    # Определяем, какую камеру инициализировать
    if register:
        global cap_register
        if cap_register is not None and cap_register.isOpened():
            return
    else:
        global cap
        if cap is not None and cap.isOpened():
            return
    
    # Создаём новую камеру
    new_cap = cv2.VideoCapture(0)
    new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    new_cap.set(cv2.CAP_PROP_FPS, 15)
    
    if not new_cap.isOpened():
        logger.error("Failed to open camera via OpenCV")
        return
    
    logger.info("OpenCV camera initialized")
    
    # Сохраняем в нужную глобальную переменную
    if register:
        cap_register = new_cap
    else:
        cap = new_cap


def get_frame() -> Optional[bytes]:
    """Захватывает и кодирует один кадр с меткой времени.
    Использует общий camera_source из run_jetson_stream() для избежания конфликтов.
    """
    global cap, camera_source

    # === Jetson ===
    if USE_JETSON_CAMERA:
        # Используем уже инициализированный camera_source из run_jetson_stream()
        if camera_source is None:
            logger.warning("Jetson camera_source not initialized, skipping frame")
            return None

        try:
            # Блокируем захват кадра для избежания гонок с WebRTC
            with cam_lock:
                img = camera_source.Capture(timeout=500)
            if img is None:
                logger.warning("Frame not captured from Jetson camera")
                return None
            frame = jetson_utils.cudaToNumpy(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            frame = cv2.flip(frame, 0)
        except Exception as e:
            logger.error(f"Jetson frame capture failed: {e}")
            return None
    # === OpenCV ===
    else:
        init_camera()
        with cam_lock:  # Защищаем cap.read()
            if cap is None or not cap.isOpened():
                logger.warning("OpenCV camera not available")
                return None
            ret, img = cap.read()
            if not ret or img is None:
                logger.warning("Failed to capture frame from camera")
                return None
            frame = img

    # Добавляем текущее время на кадр
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, str(datetime.now().time()), (10, 50), font, 1, (255, 255, 255), 2)

    # Кодируем в JPEG
    success, jpg = cv2.imencode('.jpg', frame)
    return jpg.tobytes() if success else None


def get_faces_json() -> tuple[list[Any], bool]:
    global cap, camera_source, face_app, faiss_index

    # Проверяем face_app (чтение атомарно в Python через GIL)
    if face_app is None:
        logger.error("Face analysis model not initialized")
        return [], False

    # === Захват кадра: jetson_utils или OpenCV ===
    frame = None
    capture_start = time.time()

    if USE_JETSON_CAMERA:
        # Используем общий camera_source из run_jetson_stream()
        if camera_source is None:
            logger.warning("camera_source not ready, skipping")
            return [], False

        try:
            with cam_lock:
                img = camera_source.Capture(timeout=500)
            capture_ms = (time.time() - capture_start) * 1000
            if img is not None:
                frame = jetson_utils.cudaToNumpy(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                logger.debug(f"Captured Jetson frame in {capture_ms:.0f}ms")
        except Exception as e:
            logger.warning(f"Failed to capture via jetson_utils: {e}")
    else:
        # Fallback: OpenCV
        init_camera()
        with cam_lock:  # Защищаем cap.read() от MJPEG-потока
            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                capture_ms = (time.time() - capture_start) * 1000
                if not ret:
                    frame = None
                else:
                    logger.debug(f"Captured OpenCV frame in {capture_ms:.0f}ms")
            else:
                logger.warning("OpenCV camera not available")

    if frame is None:
        logger.warning("Failed to capture frame from any source")
        return [], False

    # === Обработка кадра (одинаково для обоих источников) ===
    small_frame = cv2.resize(frame, (640, 480))
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Детекция лиц — face_app не thread-safe, берём блокировку
    detect_start = time.time()
    with face_lock:
        faces = face_app.get(rgb_frame)
    detect_ms = (time.time() - detect_start) * 1000

    if not faces:
        logger.debug(f"No faces detected in {detect_ms:.0f}ms")
        return [], True

    # Подготовка данных для MeetingManager
    detected_faces = []
    h_ratio = frame.shape[0] / 480
    w_ratio = frame.shape[1] / 640

    for face in faces:
        bbox = face.bbox.astype(int)
        query_emb = face.normed_embedding.reshape(1, -1).astype('float32')

        detected_faces.append({
            "embedding": query_emb,
            "bbox": bbox.tolist(),
            "confidence": float(face.get("confidence", 0.0)),
            "age": face.get("age"),
            "sex": face.get("sex")
        })

    # Логирование обнаруженных лиц
    logger.info(
        f"Faces detected: {len(faces)} in {detect_ms:.0f}ms | "
        f"frame: {frame.shape[1]}x{frame.shape[0]}"
    )

    # Обработка через MeetingManager
    processed_faces = meeting_manager.process_faces(detected_faces)

    # Распознавание и формирование результатов
    detections = []

    for processed_face in processed_faces:
        bbox_orig = [
            int(processed_face["bbox"][0] * w_ratio),
            int(processed_face["bbox"][1] * h_ratio),
            int(processed_face["bbox"][2] * w_ratio),
            int(processed_face["bbox"][3] * h_ratio)
        ]

        face_name = processed_face.get("face_name") or "Unknown"
        face_uuid = processed_face.get("face_uuid")
        face_type = processed_face.get("face_type")
        is_first = processed_face.get("is_first_meeting", False)
        meeting_msg = processed_face.get("meeting_message", "")

        logger.info(
            f"  Face: name={face_name}, type={face_type}, uuid={face_uuid[:8] if face_uuid else 'N/A'}, "
            f"first_meeting={is_first}, confidence={processed_face.get('confidence', 0):.2f}"
        )
        if meeting_msg:
            logger.debug(f"  Meeting: {meeting_msg}")

        detections.append(FaceDetection(
            bbox=bbox_orig,
            name=face_name,
            age=processed_face.get("age"),
            sex=processed_face.get("sex"),
            confidence=processed_face.get("confidence", 0.0)
        ))

    return detections, True


def generate_mjpeg_stream():
    """Генератор MJPEG-потока."""
    while True:
        frame = get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame + b'\r\n'
        )
        time.sleep(0.05)  # ~20 FPS


@router.on_event("startup")
async def startup_event() -> None:
    """Инициализация модели face_analysis и запуск потока (Jetson или OpenCV)."""
    global stream_thread

    update_embeddings()

    # Инициализируем camera_source ДО запуска MJPEG-потока
    if USE_JETSON_CAMERA:
        try:
            camera_source = jetson_utils.videoSource(
                "csi://0?width=1280&height=720&framerate=30"
            )
            logger.info("Jetson camera_source initialized for MJPEG and face detection")
        except Exception as e:
            logger.error(f"Failed to initialize Jetson camera_source: {e}")
            camera_source = None

        # Запуск потока для WebRTC
        stream_thread = threading.Thread(target=run_jetson_stream, daemon=True)
        stream_thread.start()
        logger.info("Jetson WebRTC stream started")
    else:
        logger.info("Fallback: using MJPEG via OpenCV")


def run_jetson_stream() -> None:
    """Фоновая трансляция через Jetson (WebRTC)."""
    global output_stream, camera_source
    port = settings.CAMERA_PORT
    if not port:
        logger.error("CAMERA_PORT is not set")
        return

    stream_url = f"webrtc://@:{port}/my_stream"
    logger.info(f"WebRTC streaming started: {stream_url}")

    while True:
        try:
            # Создаём камеру только если не инициализирована или была ошибка
            if camera_source is None:
                camera_source = jetson_utils.videoSource(
                    "csi://0?width=1280&height=720&framerate=30"
                )
                output_stream = jetson_utils.videoOutput(stream_url)
                logger.info("Jetson camera and output stream initialized")

            with cam_lock:
                img = camera_source.Capture()
            if img is None:
                logger.warning("Frame not captured from Jetson camera")
                camera_source = None  # Пересоздадим при следующей итерации
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
            camera_source = None  # Пересоздадим при следующей итерации
            output_stream = None
            time.sleep(2)


# === ЭНДПОИНТЫ ===

@router.get("/", response_class=HTMLResponse)
async def camera_page(request: Request) -> HTMLResponse:
    """
    Основная страница с видеотрансляцией.
    Показывает ссылку на WebRTC или MJPEG-поток.
    """
    server_host = request.url.hostname or "127.0.0.1"
    server_port = request.url.port or 8000

    if USE_JETSON_CAMERA:
        port = settings.CAMERA_PORT
        if not port:
            logger.error("CAMERA_PORT is not set")
            return HTMLResponse(content="Server configuration error", status_code=500)
        stream_url = f"http://{server_host}:{port}/my_stream"
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
                <img src="/video" alt="MJPEG Stream" style="max-width:100%" />
            </body>
            </html>
        """
    return HTMLResponse(content=content, status_code=200)


@router.get("/video", response_class=Response)
@router.get("/my_stream", response_class=Response)
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

    faces, status = get_faces_json()

    return DetectionResponse(
        timestamp=str(datetime.now().isoformat()),
        faces=faces,
        success=status
    )


@router.post("/register-face")
async def register_face(request: RegisterFaceRequest):
    """
    Регистрация нового лица.
    Ожидает, что на кадре ровно одно лицо.
    Делает 5 снимков, сохраняет эмбеддинги как {username}-{idx}.npy
    """
    import re

    # Валидация username
    if not re.match(r'^[a-z][a-z0-9_]{1,29}$', request.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username must be 1-30 chars: lowercase letters, digits, underscores"
        )

    username = request.username

    # Инициализация отдельной камеры для регистрации
    init_camera(register=True)
    if cap_register is None or not cap_register.isOpened():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Camera not available"
        )

    # Проверяем face_app (быстро, без блокировки)
    if face_app is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face analysis model not initialized"
        )

    # face_app не thread-safe — блокируем на время детекции
    with face_lock:
        embeddings = []
        attempts = 0
        max_attempts = 30
        required_faces = 5

        while len(embeddings) < required_faces and attempts < max_attempts:
            ret, frame = cap_register.read()
            if not ret or frame is None:
                attempts += 1
                time.sleep(0.2)
                continue

            rgb_frame = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2RGB)
            faces = face_app.get(rgb_frame)

            if len(faces) == 1:
                embedding = faces[0].normed_embedding
                embeddings.append(embedding)
                logger.info(f"Captured embedding {len(embeddings)}/{required_faces} for '{username}'")
                time.sleep(0.5)
            else:
                logger.debug(f"Skipped frame: {len(faces)} faces detected (expected exactly 1)")

            attempts += 1

    if len(embeddings) < required_faces:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not capture enough single-face frames. Got {len(embeddings)}, needed {required_faces}"
        )

    # Сохранение эмбеддингов
    saved_files = []
    embeddings_dir = settings.EMBEDDINGS_DIR
    os.makedirs(embeddings_dir, exist_ok=True)

    for idx, emb in enumerate(embeddings, start=1):
        filename = f"{username}-{idx}.npy"
        filepath = os.path.join(embeddings_dir, filename)
        np.save(filepath, emb)
        saved_files.append(filename)

    # Перезагрузка базы эмбеддингов
    update_embeddings()

    logger.info(f"Successfully registered user '{username}' with {len(saved_files)} embeddings")

    return {
        "success": True,
        "message": f"User '{username}' registered successfully",
        "embeddings_saved": saved_files,
        "count": len(saved_files)
    }


# === Глобальная инициализация (после определения всех функций) ===
data_sender = PeriodicFacesDataSender(get_faces_json, LLM_ENDPOINT_URL, 2)
data_sender.start()

# === Инициализация менеджера встреч ===
MEETING_STORAGE_PATH = settings.EMBEDDINGS_DIR.parent / "meeting_storage.json"
meeting_manager = MeetingManager(
    storage_path=MEETING_STORAGE_PATH,
    llm_endpoint_url=LLM_ENDPOINT_URL,
    detection_threshold=settings.CAMERA_DETECTION_THRESHOLD
)
meeting_manager.start()
logger.info("MeetingManager initialized")
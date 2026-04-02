from fastapi import APIRouter, Request
from starlette.responses import StreamingResponse
import threading
from datetime import datetime
import logging
import cv2
import numpy as np

from app.config.config import settings

router = APIRouter(prefix="/camera", tags=["camera"])

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальные переменные
lock = threading.Lock()
cap = None
camera = None  # для jetson.utils.gstCamera
cuda_img = None

# Попытка импорта jetson.utils
JETSON_CAMERA_AVAILABLE = False



def get_opencv_camera():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(int(settings.CAMERA_DEVICE_INDEX))
        if not cap.isOpened():
            logger.error("❌ Не удалось открыть камеру через OpenCV")
            raise RuntimeError("Не удалось открыть камеру")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        logger.info("📹 Используем OpenCV для захвата видео")
    return cap

def get_frame():
    global cuda_img
    with lock:
            try:
            cap = get_opencv_camera()
            ret, img = cap.read()
            if not ret:
                logger.warning("⚠️ Не удалось получить кадр через OpenCV")
                return None
        except Exception as e:
            logger.error(f"📷 Ошибка захвата кадра: {e}")
            return None

        # Добавляем метку времени
        font = cv2.FONT_HERSHEY_SIMPLEX
        timestamp = str(datetime.now().time())
        cv2.putText(img, timestamp, (10, 50), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # Кодируем в JPEG
        ret, jpg = cv2.imencode(".jpg", img)
        if not ret:
            logger.warning("⚠️ Не удалось закодировать изображение")
            return None

        return bytes(jpg)

def generate_video_stream():
    while True:
        frame = get_frame()
        if frame is None:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )

@router.get("/")
async def index_page(request: Request):
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory=settings.CAMERA_DOC_ROOT)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "ip": request.client.host,
            "port": request.url.port or 80
        }
    )

@router.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_video_stream(),
        media_type="multipart/x-mixed-replace;boundary=frame"
    )
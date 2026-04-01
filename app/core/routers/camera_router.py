from fastapi import APIRouter, Request, Response
from starlette.responses import StreamingResponse
import cv2
import threading
from datetime import datetime
import logging

from app.config.config import settings

router = APIRouter(prefix="/camera", tags=["camera"])

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальные переменные
lock = threading.Lock()
cap = None

def get_camera():
    global cap
    if cap is None or not cap.isOpened():
        # Попробуем использовать GStreamer pipeline для лучшей совместимости с Jetson
        pipeline = (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), width=1280, height=720, "
            "format=NV12, framerate=30/1 ! "
            "nvvidconv ! "
            "video/x-raw, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            logger.error("Cannot open camera with GStreamer pipeline")
            # Резервный вариант - прямое подключение
            cap = cv2.VideoCapture(int(settings.CAMERA_DEVICE_INDEX))
            if not cap.isOpened():
                logger.error(f"Cannot open camera at index {settings.CAMERA_DEVICE_INDEX}")
                raise RuntimeError(f"Cannot open camera at index {settings.CAMERA_DEVICE_INDEX}")
    return cap


def get_frame():
    """
    Захватывает один кадр с камеры, добавляет метку времени и кодирует в JPEG.
    """
    global cap
    with lock:
        camera = get_camera()
        ret, img = camera.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            # Попробуем пересоздать соединение с камерой
            if cap is not None:
                cap.release()
            cap = None
            return None

        # Добавляем метку времени
        font = cv2.FONT_HERSHEY_SIMPLEX
        timestamp = str(datetime.now().time())
        cv2.putText(img, timestamp, (10, 50), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Кодируем изображение в JPEG
        ret, jpg = cv2.imencode(".jpg", img)
        if not ret:
            logger.warning("Failed to encode image")
            return None

        return bytes(jpg)


def generate_video_stream():
    """
    Генерирует поток видеокадров в формате multipart/x-mixed-replace.
    """
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
    """
    Возвращает HTML-страницу с видеопотоком.
    """
    # Получаем IP клиента
    client_host = request.client.host
    port = request.url.port or 80

    # Рендерим шаблон
    from fastapi.templating import Jinja2Templates

    templates = Jinja2Templates(directory=settings.CAMERA_DOC_ROOT)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "ip": client_host,
            "port": port
        }
    )


@router.get("/video_feed")
async def video_feed():
    """
    Возвращает видеопоток в формате MJPEG.
    """
    return StreamingResponse(
        generate_video_stream(),
        media_type="multipart/x-mixed-replace;boundary=frame"
    )

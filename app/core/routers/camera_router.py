from fastapi import APIRouter, Request
from starlette.responses import StreamingResponse
from app.core.low_evel.camera import Camera, FrameReader
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
camera = Camera()  # для jetson.utils.gstCamera
frame_reader = FrameReader(camera, "nv")
frame_reader.daemon(True)
frame_reader.start()
cuda_img = None

# Попытка импорта jetson.utils
JETSON_CAMERA_AVAILABLE = False

def get_frame():
    with lock:
        try:
            ret, img = frame_reader.getFrame()
            if not ret or img is None:
                logger.warning("⚠️ Не удалось получить кадр через OpenCV")
                return None
        except Exception as e:
            logger.error(f"📷 Ошибка захвата кадра: {e}")
            return None

        # Проверяем, что изображение не пустое и не заполнено одним цветом
        if img is None or img.size == 0 or (img.ndim >= 2 and np.all(img == img[0,0])):
            logger.warning("⚠️ Получено изображение заполненное одним цветом или пустое")
            return None
            
        # Добавляем метку времени
        font = cv2.FONT_HERSHEY_SIMPLEX
        timestamp = str(datetime.now().time())
        cv2.putText(img, timestamp, (10, 50), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Дополнительная проверка: если среднее значение пикселей слишком высокое (например, все зеленое), пропускаем кадр
        if np.mean(img) > 200 and np.std(img) < 50:  # Высокое среднее и низкое стандартное отклонение
            logger.warning("⚠️ Получено подозрительное изображение (возможно, шум или ошибка)")
            return None

        # Кодируем в JPEG с качеством 80 для баланса качества и производительности
        ret, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
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
import asyncio
import logging
from datetime import datetime

import cv2
import numpy as np
from fastapi import APIRouter, Request
from starlette.responses import StreamingResponse
from app.config.config import settings

router = APIRouter(prefix="/camera", tags=["camera"])

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Попытка импорта jetson.utils
try:
    import jetson.utils
    JETSON_CAMERA_AVAILABLE = True
    logger.info("✅ Используется jetson.utils")
except ImportError as e:
    logger.error(f"❌ Не удалось импортировать jetson.utils: {e}")
    JETSON_CAMERA_AVAILABLE = False

# Глобальные переменные
camera = None

def open_camera():
    global camera
    if not JETSON_CAMERA_AVAILABLE:
        logger.error("❌ jetson.utils недоступен")
        return False
    try:
        # Используем csi://0 для первой CSI-камеры
        camera = jetson.utils.videoSource("csi://0", options={
            "width": 640,
            "height": 480,
            "framerate": 30
        })
        logger.info("📷 Камера videoSource открыта: csi://0")
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка открытия камеры: {e}")
        return False

def get_frame():
    global camera
    try:
        # Захват кадра (в формате CUDA)
        img_cuda = camera.Capture(timeout=2000)
        if img_cuda is None:
            logger.warning("⚠️ Не удалось захватить кадр (timeout)")
            return None
    except Exception as e:
        logger.error(f"📷 Ошибка захвата кадра: {e}")
        return None

    # Конвертируем CUDA -> Numpy (CPU) только при необходимости
    # Внимание: cudaToNumpy — медленная операция!
    img_cpu = jetson.utils.cudaToNumpy(img_cuda)
    img_bgr = cv2.cvtColor(img_cpu, cv2.COLOR_RGBA2BGR)

    # Фильтр: одноцветные/битые кадры
    if np.mean(img_bgr) > 200 and np.std(img_bgr) < 50:
        logger.warning("⚠️ Подозрительный кадр (возможно, шум)")
        return None

    # Добавляем метку времени
    font = cv2.FONT_HERSHEY_SIMPLEX
    timestamp = str(datetime.now().time())[:11]  # обрезаем до миллисекунд
    cv2.putText(img_bgr, timestamp, (10, 50), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # Кодируем в JPEG
    ret, jpg = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ret:
        logger.warning("⚠️ Не удалось закодировать JPEG")
        return None

    return bytes(jpg)

async def generate_video_stream():
    if not JETSON_CAMERA_AVAILABLE:
        logger.error("❌ jetson.utils недоступен, поток не запущен")
        return

    if camera is None:
        if not open_camera():
            return

    frame_interval = 1.0 / 15  # Целевые 15 FPS
    last_time = asyncio.get_event_loop().time()

    try:
        while True:
            current_time = asyncio.get_event_loop().time()
            elapsed = current_time - last_time
            if elapsed < frame_interval:
                await asyncio.sleep(frame_interval - elapsed)
            last_time = current_time

            frame = get_frame()
            if frame is None:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
    except GeneratorExit:
        logger.info("🚪 Клиент отключился")

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
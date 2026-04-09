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
    import jetson_inference_python as jetson_inference
    import jetson_utils_python as jetson_utils
except ImportError:
    import jetson_inference
    import jetson_utils


# Глобальные переменные
camera = None  # для jetson.utils.gstCamera
img_cuda = None  # текущий кадр в CUDA

def open_camera():
    global camera, img_cuda
    if not JETSON_CAMERA_AVAILABLE:
        logger.error("❌ jetson.utils недоступен")
        return False
    try:
        # Открываем камеру: 640x480, 30 FPS
        camera = camera = jetson_utils.videoSource("csi://0")
        camera.Open()
        logger.info("📷 Камера jetson.utils.gstCamera успешно открыта")
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка открытия камеры: {e}")
        return False

def get_frame():
    global camera, img_cuda
    try:
        # Захват кадра в формате RGBA (на GPU)
        img_cuda = camera.Capture()
        if img_cuda is None:
            logger.warning("⚠️ Не удалось захватить кадр (timeout)")
            return None
    except Exception as e:
        logger.error(f"📷 Ошибка захвата кадра: {e}")
        return None

    # Конвертируем CUDA -> Numpy (CPU), BGR (OpenCV)
    img_cpu = jetson.utils.cudaToNumpy(img_cuda)
    img_bgr = cv2.cvtColor(img_cpu, cv2.COLOR_RGBA2BGR)

    # Проверка: не пустое ли изображение
    if img_bgr.size == 0 or (img_bgr.ndim >= 2 and np.all(img_bgr == img_bgr[0,0])):
        logger.warning("⚠️ Получено пустое или одноцветное изображение")
        return None

    # Фильтр подозрительных кадров (например, полностью белое/зелёное)
    mean_val = np.mean(img_bgr)
    std_val = np.std(img_bgr)
    if mean_val > 200 and std_val < 50:
        logger.warning("⚠️ Подозрительный кадр: высокая яркость, низкая вариация")
        return None

    # Добавляем метку времени
    font = cv2.FONT_HERSHEY_SIMPLEX
    timestamp = str(datetime.now().time())
    cv2.putText(img_bgr, timestamp, (10, 50), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # Кодируем в JPEG
    ret, jpg = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ret:
        logger.warning("⚠️ Не удалось закодировать изображение в JPEG")
        return None

    return bytes(jpg)

async def generate_video_stream():
    if not JETSON_CAMERA_AVAILABLE:
        logger.error("❌ jetson.utils недоступен, поток не запущен")
        return

    if camera is None:
        if not open_camera():
            return

    try:
        while True:
            frame = get_frame()
            if frame is None:
                await asyncio.sleep(0.01)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            await asyncio.sleep(0.01)  # даем шанс на отключение
    except GeneratorExit:
        logger.info("🚪 Клиент отключился от видеопотока")

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
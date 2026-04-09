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
    try:
        # Открываем камеру: 640x480, 30 FPS
        camera = jetson_utils.videoSource("csi://0", argv=["--input-width=640", "--input-height=360"])
        camera.Open()
        logger.info("📷 Камера jetson.utils.gstCamera успешно открыта")
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка открытия камеры: {e}")
        return False


def get_frame():
    global camera
    # Захватываем кадр. Jetson автоматически держит его в CUDA.
    img_cuda = camera.Capture()
    if img_cuda is None:
        return None

    # ВАЖНО: Вместо cv2.imencode используйте аппаратное сохранение в память (если библиотека позволяет)
    # Или хотя бы уменьшите частоту проверок (std_val, mean_val), они очень медленные на CPU.

    img_cpu = jetson_utils.cudaToNumpy(img_cuda)
    # Уменьшаем размер кадра перед обработкой OpenCV в 2 раза
    img_cpu = cv2.resize(img_cpu, (320, 240))
    img_bgr = cv2.cvtColor(img_cpu, cv2.COLOR_RGBA2BGR)

    # Кодируем с минимальным качеством для скорости
    ret, jpg = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return bytes(jpg)

async def generate_video_stream():

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
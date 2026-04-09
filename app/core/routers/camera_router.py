import logging
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from app.config.config import settings
import jetson_utils
import threading

router = APIRouter(prefix="/camera", tags=["camera"])
templates = Jinja2Templates(directory=settings.CAMERA_DOC_ROOT)

# Глобальные объекты
output_stream = None
camera_source = None
stream_thread = None


def run_stream():
    """Фоновый поток для захвата и трансляции"""
    global output_stream, camera_source

    # Инициализация (WebRTC сервер запустится на порту 8554 по умолчанию)
    camera_source = jetson_utils.videoSource("csi://0")
    output_stream = jetson_utils.videoOutput("webrtc://@:8554/my_stream")

    while True:
        img = camera_source.Capture()
        if img is None:
            continue
        output_stream.Render(img)
        output_stream.SetStatus(f"Streaming @ {output_stream.GetFrameRate():.1f} FPS")


@router.on_event("startup")
async def startup_event():
    global stream_thread
    stream_thread = threading.Thread(target=run_stream, daemon=True)
    stream_thread.start()
    logging.info("🚀 WebRTC сервер запущен на порту 8554")


@router.get("/")
async def index_page(request: Request):
    # Передаем IP и порт WebRTC в шаблон
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "stream_url": f"http://{request.client.host}:8554/my_stream"
        }
    )

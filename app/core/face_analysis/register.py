import logging
import sys
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import os

# Добавляем корень проекта
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from app.core.face_analysis.core import get_pipeline
from config.config import settings

# Настройка логгера
logger = logging.getLogger("register_logger")
camera_source: Optional['jetson_utils.videoSource'] = None

USE_JETSON_CAMERA = False
try:
    import jetson_utils
    USE_JETSON_CAMERA = True
except ImportError:
    logger.warning("Module jetson_utils not found. Using OpenCV fallback.")

cap = None
camera_initialized = False

def init_camera() -> None:
    global cap, camera_source
    if USE_JETSON_CAMERA:
        if camera_source is None:
            camera_source = jetson_utils.videoSource("csi://0")
        return
    """Ленивая инициализация OpenCV камеры."""
    global cap, camera_initialized
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

    if USE_JETSON_CAMERA:
        img = camera_source.Capture(timeout=1000)
    else:
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

def register_face():
    app = get_pipeline()
    os.makedirs(settings.EMBEDDINGS_DIR, exist_ok=True)

    name = input("Введите имя человека: ")
    cap = cv2.VideoCapture(0)

    print("Нажмите 'S' для захвата, 'Q' для выхода")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр с камеры.")
            break

        faces = app.get(frame)
        img_draw = frame.copy()

        for face in faces:
            box = face.bbox.astype(int)
            cv2.rectangle(img_draw, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)

        cv2.imshow("Register", img_draw)
        key = cv2.waitKey(1)

        np.save(f"{settings.EMBEDDINGS_DIR}/{name}.npy", faces[0].normed_embedding)
        print(f"Готово! Лицо {name} сохранено.")
        break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    register_face()
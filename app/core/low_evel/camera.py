# MIT License
# Copyright (c) 2019 JetsonHacks
# See license
# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV

import time
import cv2

from app.config.config import settings

try:
    from Queue import Queue
except ModuleNotFoundError:
    from queue import Queue

import threading


def gstreamer_pipeline():
    return (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
        "nvvideoconvert ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink max-buffers=1 drop=true"
    )


class FrameReader(threading.Thread):
    def __init__(self, cap, name="frame_reader"):
        super().__init__()
        self.name = name
        self.cap = cap
        self.queue = Queue(maxsize=1)
        self._running = True
        self.daemon = True
        self.start()

    def run(self):
        while self._running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                if not self.queue.full():
                    try:
                        self.queue.put_nowait(frame)
                    except Queue.Full:
                        pass  # Пропускаем, если очередь полна — это нормально при высокой нагрузке
            else:
                print("⚠️ Не удалось получить кадр из камеры")
                time.sleep(0.01)
        # Освобождаем камеру при завершении потока
        self.cap.release()

    def getFrame(self, timeout=0.1):  # Сократили таймаут
        try:
            return self.queue.get_nowait()  # Попробуем без блокировки
        except Queue.Empty:
            pass
        try:
            return self.queue.get(timeout=timeout)
        except Queue.Empty:
            return None
        except Exception as e:
            print(f"Ошибка получения кадра: {e}")
            return None

    def stop(self):
        self._running = False
        self.queue.queue.clear()


class Camera:
    def __init__(self):
        self.cap = None
        self.frame_reader = None
        self.open_camera()

    def open_camera(self):
        # Останавливаем предыдущий поток, если он существует
        if self.frame_reader is not None:
            self.frame_reader.stop()
            self.frame_reader.join(timeout=1)

        # Открываем камеру с GStreamer
        self.cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError(
                "Failed to open camera with GStreamer pipeline. "
                "Check sensor-id, camera connection, or nvargus-daemon."
            )

        # Создаём новый FrameReader
        self.frame_reader = FrameReader(self.cap)

    def getFrame(self):
        # Убрали sleep — не нужно искусственно замедлять
        frame = self.frame_reader.getFrame(timeout=0.5)  # 500ms максимум ждём
        if frame is not None:
            return True, frame.copy()  # Возвращаем копию, чтобы избежать проблем с памятью
        else:
            return False, None

    def close(self):
        if self.frame_reader is not None:
            self.frame_reader.stop()
            self.frame_reader.join(timeout=1)
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        print("Camera closed.")


if __name__ == "__main__":
    camera = Camera()
    print("Camera opened. Reading frames for 10 seconds...")
    start_time = time.time()
    while time.time() - start_time < 10:
        ret, frame = camera.getFrame()
        if ret:
            print("✅ Кадр получен")
            # cv2.imwrite("test_frame.jpg", frame)  # При необходимости сохранить
        else:
            print("❌ Нет кадра")
        time.sleep(0.1)
    camera.close()
# MIT License
# Copyright (c) 2019 JetsonHacks
# See license
# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image

import time
import cv2

from app.config.config import settings

try:
    from Queue import Queue
except ModuleNotFoundError:
    from queue import Queue

import threading


# GStreamer pipeline — используем nvarguscamerasrc для CSI-камеры
def gstreamer_pipeline():
    return (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
        "nvvideoconvert ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink max-buffers=1 drop=true"
    )


class FrameReader(threading.Thread):
    def __init__(self, camera, name):
        threading.Thread.__init__(self)
        self.name = name
        self.camera = camera
        self.queue = Queue(1)
        self._running = True
        self.daemon = True
        self.start()

    def run(self):
        while self._running:
            ret, frame = self.camera.getFrame()
            print(f"Получили фрейм ret={ret} frame={frame}")
            if ret and frame is not None:
                if not self.queue.full():
                    self.queue.put(frame)
                    print("фреймы в очереди")
            time.sleep(0.01)  # Небольшая задержка для снижения нагрузки на CPU

    def getFrame(self, timeout=None):
        return self.queue.get(timeout=timeout)

    def stop(self):
        self._running = False
        self.queue.queue.clear()  # Очищаем очередь при остановке


class Camera(object):
    cap = None

    def __init__(self):
        self.cap = None
        self.frame_reader = None
        self.open_camera()
        
    def open_camera(self):
        # Останавливаем предыдущий поток, если он существует
        if self.frame_reader is not None:
            self.frame_reader.stop()
            self.frame_reader.join(timeout=1)
        
        # Используем GStreamer-пайплайн
        self.cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError(
                "Failed to open camera with GStreamer pipeline. "
                "Check sensor-mode, camera connection, or nvargus-daemon."
            )
        
        # Создаем и запускаем поток для чтения кадров
        self.frame_reader = FrameReader(self.cap, "frame_reader")

    def open_camera(self):
        # Используем GStreamer-пайплайн
        self.cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError(
                "Failed to open camera with GStreamer pipeline. "
                "Check sensor-mode, camera connection, or nvargus-daemon."
            )

    def getFrame(self):
        if self.frame_reader and self.frame_reader.is_alive():
            try:
                frame = self.frame_reader.getFrame(timeout=2.0)
                return (True, frame) if frame is not None else (False, None)
            except Exception as e:
                print(f"Frame reader error: {e}")
                return (False, None)
        return (False, None)

    def close(self):
        self.cap.release()


if __name__ == "__main__":
    camera = Camera()
    print("Camera opened. Reading frames for 10 seconds...")
    start_time = time.time()
    while time.time() - start_time < 10:
        ret, frame = camera.getFrame()
        if ret:
            print("✅ Кадр получен")
            # Можно сохранить: cv2.imwrite("test_frame.jpg", frame)
        time.sleep(0.1)
    camera.close()
    print("Camera closed.")
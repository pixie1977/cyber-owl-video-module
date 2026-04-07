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
        "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=60/1 ! "
        "nvvideoconvert ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink"
    )


class FrameReader(threading.Thread):
    queues = []
    _running = True
    camera = None

    def __init__(self, camera, name):
        threading.Thread.__init__(self)
        self.name = name
        self.camera = camera

    def run(self):
        while self._running:
            ret, frame = self.camera.read()
            if not ret:
                continue
            while self.queues:
                queue = self.queues.pop()
                if not queue.full():
                    queue.put(frame)

    def addQueue(self, queue):
        self.queues.append(queue)

    def getFrame(self, timeout=None):
        queue = Queue(1)
        self.addQueue(queue)
        return queue.get(timeout=timeout)

    def stop(self):
        self._running = False


class Camera(object):
    cap = None

    def __init__(self):
        self.open_camera()

    def open_camera(self):
        # Используем GStreamer-пайплайн
        self.cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError(
                "Failed to open camera with GStreamer pipeline. "
                "Check sensor-mode, camera connection, or nvargus-daemon."
            )

    def getFrame(self):
        return self.cap.read()  # (ret, frame)

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
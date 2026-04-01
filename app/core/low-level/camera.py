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
            while self.queues:
                queue = self.queues.pop()
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
    # frame_reader = None
    cap = None

    def __init__(self):
        self.open_camera()

    def open_camera(self):
        self.cap = cv2.VideoCapture(settings.CAMERA_DEVICE_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera!")

        # if self.frame_reader == None:
        #     self.frame_reader = FrameReader(self.cap, "")
        #     self.frame_reader.daemon = True
        #     self.frame_reader.start()

    def getFrame(self):
        return self.cap.read()

    def close(self):
        # self.frame_reader.stop()
        self.cap.release()


if __name__ == "__main__":
    camera = Camera()
    camera.start_preview()
    time.sleep(10)
    camera.stop_preview()
    camera.close()

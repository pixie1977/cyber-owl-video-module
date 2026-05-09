import sys

import cv2
import numpy as np
import os

# Добавляем корень проекта
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

# Теперь можно импортировать app и config
# Изменён импорт: теперь абсолютный
from app.core.face_analysis.core import get_pipeline
from config.config import settings


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

        if key & 0xFF == ord('s') and len(faces) == 1:
            np.save(f"{settings.EMBEDDINGS_DIR}/{name}.npy", faces[0].normed_embedding)
            print(f"Готово! Лицо {name} сохранено.")
            break
        elif key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    register_face()
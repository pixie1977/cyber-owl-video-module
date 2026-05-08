import cv2
import faiss
import numpy as np
import glob
import os

# Абсолютный импорт
from app.core.face_analysis.core import get_pipeline
from config.config import settings

db_names = []
db_embeddings = []

def update_embeddings():
    global db_names
    global db_embeddings
    for file in glob.glob("embeddings/*.npy"):
        db_names.append(os.path.basename(file).replace(".npy", ""))
        db_embeddings.append(np.load(file))

def main():
    # 1. Загрузка модели и базы данных
    app = get_pipeline()

    update_embeddings()

    if not db_embeddings:
        print("База пуста! Сначала запустите register.py")
        return

    # 2. Создание FAISS индекса
    dim = db_embeddings[0].shape[0]  # например, 512
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(db_embeddings).astype('float32'))

    # 3. Настройка камеры
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Уменьшаем разрешение
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Для скорости
    cap.set(cv2.CAP_PROP_FPS, 15)            # Ограничиваем FPS

    frame_skip = 2  # Пропускать 2 кадра из 3
    frame_count = 0

    print("Нажмите 'Q' для выхода")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Не удалось получить кадр.")
                break

            frame_count += 1
            # Пропускаем кадры для ускорения
            if frame_count % (frame_skip + 1) != 0:
                cv2.imshow("SOTA Face ID", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Масштабирование кадра (необязательно, но ускоряет)
            small_frame = cv2.resize(frame, (640, 480))
            faces = app.get(small_frame)

            for face in faces:
                bbox = face.bbox.astype(int)
                # Восстанавливаем координаты, если нужно (не нужно, если не масштабировали)
                query_emb = face.normed_embedding.reshape(1, -1).astype('float32')

                distances, indices = index.search(query_emb, k=1)
                score = distances[0][0]
                idx = indices[0][0]

                if score > settings.CAMERA_DETECTION_THRESHOLD:
                    name = f"{db_names[idx]} ({score:.2f})"
                    color = (0, 255, 0)
                else:
                    name = "Unknown"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(frame, name, (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("SOTA Face ID", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
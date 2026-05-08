import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os

# 1. Инициализация SOTA модели
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))


def capture_and_save():
    name = input("Введите имя человека для регистрации: ").strip()
    if not name:
        print("Имя не может быть пустым.")
        return

    cap = cv2.VideoCapture(0)
    print("\nИнструкция:")
    print("- Смотрите в камеру")
    print("- Нажмите 'S' для захвата и сохранения")
    print("- Нажмите 'Q' для выхода без сохранения\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Только детекция для предпросмотра (быстро)
        faces = app.get(frame)

        display_frame = frame.copy()

        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(display_frame, "READY TO CAPTURE", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        cv2.imshow("Registration - Press 'S' to Save", display_frame)

        key = cv2.waitKey(1) & 0xFF

        # Сохранение по нажатию 'S'
        if key == ord('s') or key == ord('ы'):
            if len(faces) == 0:
                print("Ошибка: Лицо не найдено! Попробуйте еще раз.")
            elif len(faces) > 1:
                print("Ошибка: В кадре больше одного лица!")
            else:
                # Берем эмбеддинг первого найденного лица
                embedding = faces[0].normed_embedding

                # Создаем папку, если её нет
                if not os.path.exists("embeddings"):
                    os.makedirs("embeddings")

                # Сохраняем вектор в формате numpy
                file_path = f"embeddings/{name}.npy"
                np.save(file_path, embedding)

                print(f"Успешно! Эмбеддинг для '{name}' сохранен в {file_path}")
                break

        elif key == ord('q') or key == ord('й'):
            print("Отменено пользователем.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_and_save()

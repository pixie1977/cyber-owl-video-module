import cv2
import mediapipe as mp

# --- Настройка MediaPipe Face Detection ---
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# model_selection=0: дальнее расстояние (до 5 м)
# min_detection_confidence: порог уверенности (снижено для теста)
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.3
)

# --- Инициализация камеры ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Камера не доступна")
    exit(1)

# Устанавливаем разрешение
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("🚀 Запуск камеры. Нажмите 'q' для выхода.")
print("💡 Повернитесь к камере. Должны появиться рамки и точки.")

frame_count = 0

while True:
    ret, frame = cap.read()
    frame_count += 1
    if not ret:
        print("❌ Не удалось получить кадр")
        break

    if frame_count % 2 == 0:  # Пропускаем каждый 2-й кадр для плавности
        continue

    # Конвертируем в RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False  # Опция для производительности

    # Обработка
    results = face_detection.process(rgb_frame)

    # Обратно в BGR
    rgb_frame.flags.writeable = True
    frame_copy = frame.copy()

    # Если найдены лица
    if results.detections:
        print(f"✅ Лицо обнаружено: {len(results.detections)}")

        for detection in results.detections:
            # Рисуем стандартные аннотации (рамка + ключевые точки)
            mp_drawing.draw_detection(frame_copy, detection)

    else:
        # Для отладки: яркость кадра
        brightness = cv2.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))[0]
        if brightness < 20:
            status = "СЛИШКОМ ТЁМНО"
            color = (0, 0, 255)
        elif brightness > 230:
            status = "ПЕРЕСВЕТ"
            color = (0, 0, 255)
        else:
            status = "НЕТ ЛИЦ"
            color = (0, 255, 255)

        cv2.putText(frame_copy, status, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('MediaPipe Face Detection Test', frame_copy)

    # Выход по 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Очистка ---
cap.release()
cv2.destroyAllWindows()
face_detection.close()
print("👋 Тест завершён.")
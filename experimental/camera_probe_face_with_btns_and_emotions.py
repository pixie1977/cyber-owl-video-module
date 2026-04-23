# experimental/camera_probe_face_with_btns_and_emotions.py
import cv2
import os
import numpy as np
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS
from scipy.optimize import linear_sum_assignment
import mediapipe as mp
import onnxruntime as ort

# --- Настройки ---
DATASET_PATH = "face_dataset"
FACE_MODEL_PATH = os.path.join("models", "face_detection")
FACE_MODEL_PATH = os.path.join(FACE_MODEL_PATH, "face_recognition_sface_2021dec.onnx")
FACE_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

EMOTION_MODEL_PATH = "models/emotion/emotion-ferplus-12-int8.onnx"
EMOTION_URL = "https://huggingface.co/microsoft/emotion-ferplus/resolve/main/emotion-ferplus-8.onnx?download=true"
CONFIDENCE_THRESHOLD = 0.363
SKIP_FRAMES = 5
MAX_AGE = 10

# Создаём папки
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(os.path.dirname(EMOTION_MODEL_PATH), exist_ok=True)

# --- Функция загрузки файла ---
def download_file(url, path, name):
    if not os.path.exists(path):
        print(f"⏬ Скачиваем {name}...")
        try:
            import requests
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # Проверка размера после загрузки
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            size_mb = os.path.getsize(path) / (1024 * 1024)
            if size_mb < 1.0:
                print(f"❌ Подозрительный размер: {size_mb:.2f} МБ — возможно, HTML вместо модели")
                os.remove(path)
                exit(1)

            print(f"✅ {name} скачана: {size_mb:.1f} МБ")
        except Exception as e:
            print(f"❌ Ошибка загрузки {name}: {e}")
            exit(1)

# Скачиваем модели
download_file(FACE_MODEL_URL, FACE_MODEL_PATH, "SFace модель")
download_file(EMOTION_URL, EMOTION_MODEL_PATH, "модель эмоций")

# --- Загрузка SFace ---
try:
    recognizer = cv2.FaceRecognizerSF.create(FACE_MODEL_PATH, "")
except Exception as e:
    print(f"❌ Не удалось загрузить SFace: {e}")
    exit(1)

# --- Загрузка модели эмоций ---
try:
    emotion_session = ort.InferenceSession(EMOTION_MODEL_PATH)
    emotion_labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'angry', 'neutral']
    print("✅ Модель эмоций загружена")
except Exception as e:
    print(f"❌ Не удалось загрузить модель эмоций: {e}")
    exit(1)

# --- Загрузка известных лиц ---
embeddings = []
labels = []

def load_embeddings():
    global embeddings, labels
    embeddings.clear()
    labels.clear()
    for root, _, files in os.walk(DATASET_PATH):
        for file in files:
            name, ext = os.path.splitext(file.lower())
            if ext in [".jpg", ".jpeg", ".png", ".bmp"] and not file.startswith(("unknown_", "temp_")):
                path = os.path.join(root, file)
                img = cv2.imread(path)
                if img is None:
                    continue
                h, w = img.shape[:2]
                x, y, w_face, h_face = int(w * 0.2), int(h * 0.2), int(w * 0.6), int(h * 0.6)
                face_aligned = recognizer.alignCrop(img, (x, y, w_face, h_face))
                if face_aligned is None:
                    continue
                embedding = recognizer.feature(face_aligned)
                embeddings.append(embedding)
                labels.append(f"known_{name}")
    print(f"✅ Загружено: {len(embeddings)} шаблонов лиц")

load_embeddings()

# --- MediaPipe Face Detection ---
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# --- Камера ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("🚀 Распознавание лиц + эмоции. Клавиши: [S] Save, [R] Reload, [Q] Quit")

# --- Трекинг лиц ---
tracked_faces = {}
next_track_id = 0
frame_count = 0

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = w1 * h1
    area2 = w2 * h2
    return inter_area / (area1 + area2 - inter_area)

# --- Предсказание эмоции ---
def predict_emotion(face_crop):
    try:
        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (64, 64))
        input_data = face_resized.reshape(1, 1, 64, 64).astype(np.float32)
        input_name = emotion_session.get_inputs()[0].name
        pred = emotion_session.run(None, {input_name: input_data})[0][0]
        emotion_idx = np.argmax(pred)
        return emotion_labels[emotion_idx]
    except Exception as e:
        return "?"

# --- Отрисовка UI ---
def draw_ui(frame, status_msg=""):
    h, w = frame.shape[:2]
    # Фон панели
    cv2.rectangle(frame, (0, 0), (w, 50), (30, 30, 30), -1)

    # Кнопки
    def draw_button(label, x, color):
        cv2.rectangle(frame, (x, 10), (x + 100, 40), color, -1)
        cv2.putText(frame, label, (x + 12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    draw_button("[S] Save", 10, (0, 150, 0))      # Зелёная
    draw_button("[R] Reload", 120, (150, 150, 0)) # Жёлтая
    draw_button("[Q] Quit", 230, (0, 0, 150))     # Красная

    # Статус
    if status_msg:
        cv2.putText(frame, status_msg, (350, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

# --- Основной цикл ---
status_message = ""
while True:
    ret, frame = cap.read()
    if not ret:
        status_message = "❌ Ошибка камеры"
        break

    frame_count += 1
    process_this_frame = (frame_count % SKIP_FRAMES == 0)
    current_detections = []

    # Обнаружение лиц
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        h, w = frame.shape[:2]
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            width = int(bboxC.width * w)
            height = int(bboxC.height * h)
            current_detections.append((x, y, width, height))

    # --- Сопоставление с треками ---
    matched_tracks = {}
    current_track_ids = set()

    if tracked_faces and current_detections:
        cost_matrix = np.ones((len(tracked_faces), len(current_detections)))
        for i, track_id in enumerate(tracked_faces):
            tbox = tracked_faces[track_id]["bbox"]
            for j, dbox in enumerate(current_detections):
                cost_matrix[i, j] = 1 - compute_iou(tbox, dbox)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 0.5:
                track_id = list(tracked_faces.keys())[i]
                matched_tracks[j] = track_id
                current_track_ids.add(track_id)

    # --- Обработка каждого обнаруженного лица ---
    for idx, det_box in enumerate(current_detections):
        x, y, w, h = det_box
        name = "Unknown"
        color = (255, 0, 0)  # синий — детекция
        emotion = ""

        if idx in matched_tracks:
            track_id = matched_tracks[idx]
            track = tracked_faces[track_id]

            if process_this_frame:
                face_aligned = recognizer.alignCrop(frame, det_box)
                if face_aligned is not None:
                    emb = recognizer.feature(face_aligned)
                    scores = [recognizer.match(emb, saved_emb, cv2.FaceRecognizerSF_FR_COSINE) for saved_emb in embeddings]
                    best_score = max(scores) if scores else 0.0

                    if best_score >= CONFIDENCE_THRESHOLD:
                        best_idx = np.argmax(scores)
                        name = labels[best_idx].split("_", 1)[1]
                        color = (0, 255, 0)  # зелёный
                    else:
                        name = "Unknown"
                        color = (0, 0, 255)  # красный

                    # 🎭 Определяем эмоцию
                    face_roi = frame[y:y+h, x:x+w]
                    if face_roi.size > 0:
                        emotion = predict_emotion(face_roi)
                        tracked_faces[track_id]["emotion"] = emotion
                else:
                    name = track["name"]
                    color = track["color"]
                    emotion = track.get("emotion", "")
            else:
                name = track["name"]
                color = track["color"]
                emotion = track.get("emotion", "")

            tracked_faces[track_id]["bbox"] = det_box
            tracked_faces[track_id]["name"] = name
            tracked_faces[track_id]["color"] = color
            tracked_faces[track_id]["age"] = 0
        else:
            # Новый трек
            new_id = next_track_id
            tracked_faces[new_id] = {
                "bbox": det_box,
                "name": name,
                "color": color,
                "emotion": "",
                "age": 0
            }
            next_track_id += 1

    # Удаление старых треков
    for tid in list(tracked_faces.keys()):
        if tid not in current_track_ids:
            tracked_faces[tid]["age"] += 1
            if tracked_faces[tid]["age"] > MAX_AGE:
                del tracked_faces[tid]

    # --- Отрисовка ---
    for data in tracked_faces.values():
        x, y, w, h = data["bbox"]
        name = data["name"]
        color = data["color"]
        emotion = data.get("emotion", "")

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, name, (x, y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if emotion:
            cv2.putText(frame, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    # --- Обработка клавиш ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and len(current_detections) == 1:
        x, y, w, h = current_detections[0]
        crop = frame[y:y+h, x:x+w].copy()

        # Генерация имени
        existing = [f for f in os.listdir(DATASET_PATH) if f.startswith("person_") and f.lower().endswith((".jpg", ".jpeg", ".png"))]
        indices = [int(f.split('_')[1].split('.')[0]) for f in existing if f.split('_')[1].isdigit()]
        next_idx = max(indices) + 1 if indices else 1
        filename = f"person_{next_idx:03d}.jpg"
        filepath = os.path.join(DATASET_PATH, filename)

        cv2.imwrite(filepath, crop)
        status_message = f"📸 Сохранено: {filename}"

        face_aligned = recognizer.alignCrop(frame, (x, y, w, h))
        if face_aligned is not None:
            emb = recognizer.feature(face_aligned)
            embeddings.append(emb)
            labels.append(f"known_person_{next_idx:03d}")
            status_message += " | ✅ Добавлено в базу"
        else:
            status_message += " | ⚠️ Не выровнено"
    elif key == ord('r'):
        load_embeddings()
        status_message = f"🔁 База обновлена: {len(embeddings)} лиц"

    # Рисуем интерфейс
    draw_ui(frame, status_message)
    cv2.imshow('Cyber Owl - Face + Emotions', frame)

# --- Очистка ---
cap.release()
cv2.destroyAllWindows()
face_detection.close()
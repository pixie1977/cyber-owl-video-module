import cv2
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
import mediapipe as mp

# --- Настройки ---
DATASET_PATH = "face_dataset"
MODEL_PATH = os.path.join(DATASET_PATH, "face_recognition_sface_2021dec.onnx")
MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
CONFIDENCE_THRESHOLD = 0.363  # ✅ Рекомендуемый порог для COSINE
SKIP_FRAMES = 5
MAX_AGE = 10  # кадров без обновления → удалить
AUTO_ENROLL = True  # ✅ Авто-добавление новых лиц
ENROLL_INTERVAL = 30  # кадров между возможным добавлением

# --- Создание папки ---
os.makedirs(DATASET_PATH, exist_ok=True)

# --- Загрузка модели ---
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("⏬ Скачиваем модель SFace...")
        try:
            import requests
            response = requests.get(MODEL_URL, stream=True, timeout=30)
            response.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            if size_mb < 5.0:
                print(f"❌ Подозрительно маленький размер: {size_mb:.2f} МБ")
                os.remove(MODEL_PATH)
                exit(1)
            print(f"✅ Модель скачана: {size_mb:.1f} МБ")
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            exit(1)

download_model()

# --- Загрузка SFace ---
try:
    recognizer = cv2.FaceRecognizerSF.create(MODEL_PATH, "")
except Exception as e:
    print(f"❌ Не удалось загрузить SFace: {e}")
    exit(1)

# --- Загрузка известных лиц ---
embeddings = []
labels = []

def load_embeddings():
    global embeddings, labels
    embeddings = []
    labels = []
    for root, _, files in os.walk(DATASET_PATH):
        for file in files:
            name, ext = os.path.splitext(file.lower())
            if ext in [".jpg", ".jpeg", ".png", ".bmp"] and not file.startswith("unknown_"):
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
    print(f"✅ Загружено: {len(embeddings)} шаблонов")

load_embeddings()

# --- MediaPipe ---
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# --- Камера ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("🚀 Распознавание + авто-обучение. 'q' — выход.")

# --- Трекинг ---
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

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Проблема с кадром")
        break

    frame_count += 1
    process_this_frame = (frame_count % SKIP_FRAMES == 0)
    current_detections = []

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

    # --- Обновление треков ---
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

    # Обновляем или создаём
    for idx, det_box in enumerate(current_detections):
        x, y, w, h = det_box
        name = "Unknown"
        color = (255, 0, 0)  # синий — детекция
        crop = None

        if idx in matched_tracks:
            track_id = matched_tracks[idx]
            track = tracked_faces[track_id]

            if process_this_frame:
                face_aligned = recognizer.alignCrop(frame, det_box)
                if face_aligned is not None:
                    emb = recognizer.feature(face_aligned)
                    crop = frame[y:y+h, x:x+w].copy()
                    scores = [recognizer.match(emb, e, cv2.FaceRecognizerSF_FR_COSINE) for e in embeddings]
                    best_score = max(scores) if scores else 0.0

                    if best_score >= CONFIDENCE_THRESHOLD:
                        best_idx = np.argmax(scores)
                        name = labels[best_idx].split("_", 1)[1]
                        color = (0, 255, 0)
                    else:
                        name = "Unknown"
                        color = (0, 0, 255)

                        # ✅ Авто-обучение: сохраняем новое лицо
                        if AUTO_ENROLL and frame_count % ENROLL_INTERVAL == 0:
                            uid = f"unknown_{len(embeddings)}"
                            filename = f"{uid}.jpg"
                            filepath = os.path.join(DATASET_PATH, filename)
                            cv2.imwrite(filepath, crop)
                            print(f"📸 Добавлено новое лицо: {filename}")
                            # Обновляем базу
                            embeddings.append(emb)
                            labels.append(f"known_{uid}")
                else:
                    name = track["name"]
                    color = track["color"]
            else:
                name = track["name"]
                color = track["color"]

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
                "age": 0
            }
            next_track_id += 1

    # Удаление старых
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
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('SFace + AutoEnroll', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Очистка ---
cap.release()
cv2.destroyAllWindows()
face_detection.close()
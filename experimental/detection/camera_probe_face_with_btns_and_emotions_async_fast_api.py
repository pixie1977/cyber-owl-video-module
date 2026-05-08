# experimental/camera_probe_face_with_btns_and_emotions_async_fast_api.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import cv2
import os
import numpy as np
import threading
import queue
from scipy.optimize import linear_sum_assignment
import mediapipe as mp
import onnxruntime as ort
import time
from contextlib import asynccontextmanager

# --- Настройки ---
DATASET_PATH = "face_dataset"
FACE_MODEL_PATH = os.path.join("models", "face_detection", "face_recognition_sface_2021dec.onnx")
FACE_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

EMOTION_MODEL_PATH = "models/emotion/emotion-ferplus-8.onnx"
EMOTION_URL = "https://huggingface.co/microsoft/emotion-ferplus/resolve/main/emotion-ferplus-8.onnx?download=true"

CONFIDENCE_THRESHOLD = 0.4
MAX_AGE = 10

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(os.path.dirname(FACE_MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(EMOTION_MODEL_PATH), exist_ok=True)


# --- Загрузка файлов ---
def download_file(url, path, name):
    if not os.path.exists(path):
        print(f"⏬ Скачиваем {name}...")
        try:
            import requests
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, stream=True, timeout=60, headers=headers)
            response.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            if size_mb < 1.0:
                print(f"❌ Подозрительный размер: {size_mb:.2f} МБ")
                os.remove(path)
                exit(1)
            print(f"✅ {name} скачана: {size_mb:.1f} МБ")
        except Exception as e:
            print(f"❌ Ошибка загрузки {name}: {e}")
            exit(1)

download_file(FACE_MODEL_URL, FACE_MODEL_PATH, "SFace модель")
download_file(EMOTION_URL, EMOTION_MODEL_PATH, "модель эмоций")


# --- Инициализация моделей ---
try:
    recognizer = cv2.FaceRecognizerSF.create(FACE_MODEL_PATH, "")
except Exception as e:
    print(f"❌ Не удалось загрузить SFace: {e}")
    exit(1)

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
                if face_aligned is not None:
                    embedding = recognizer.feature(face_aligned)
                    embeddings.append(embedding)
                    labels.append(f"known_{name}")
    print(f"✅ Загружено: {len(embeddings)} шаблонов лиц")


load_embeddings()


# --- MediaPipe ---
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)


# --- Асинхронный захват кадров ---
class FrameCaptureThread(threading.Thread):
    def __init__(self, src=0, width=640, height=480):
        super().__init__()
        self.src = src
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.q = queue.Queue(maxsize=2)
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                if not self.q.empty():
                    try:
                        self.q.get_nowait()  # только последний кадр
                    except queue.Empty:
                        pass
                self.q.put((ret, frame))
            else:
                self.running = False
        self.cap.release()

    def read(self):
        try:
            return self.q.get(timeout=1)
        except queue.Empty:
            return False, None

    def stop(self):
        self.running = False


# --- Обработка лиц в отдельном потоке ---
class FaceProcessingThread(threading.Thread):
    def __init__(self, frame_queue, result_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.running = True

    def run(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                if frame is None:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)

                detections = []
                if results.detections:
                    h, w = frame.shape[:2]
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        x = int(bboxC.xmin * w)
                        y = int(bboxC.ymin * h)
                        width = int(bboxC.width * w)
                        height = int(bboxC.height * h)
                        detections.append((x, y, width, height))

                self.result_queue.put(detections)
                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Ошибка в потоке обработки: {e}")

    def stop(self):
        self.running = False


# --- Глобальный буфер для API ---
latest_faces_data = []
data_lock = threading.Lock()


def update_faces_data(tracked_faces):
    """Обновляет глобальные данные для API + смещение от центра"""
    global latest_faces_data
    faces = []
    frame_width = 640
    frame_height = 480
    center_x = frame_width / 2
    center_y = frame_height / 2

    with data_lock:
        for data in tracked_faces.values():
            x, y, w, h = data["bbox"]
            face_center_x = x + w / 2
            face_center_y = y + h / 2
            dx = float(face_center_x - center_x)  # положительное → право
            dy = float(face_center_y - center_y)  # положительное → вниз

            faces.append({
                "id": data["name"],
                "emotion": data.get("emotion", ""),
                "bbox": {
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h)
                },
                "offset_from_center": {
                    "dx": round(dx, 2),
                    "dy": round(dy, 2)
                }
            })
        latest_faces_data = faces


# --- Трекинг ---
tracked_faces = {}
next_track_id = 0


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


def predict_emotion(face_crop):
    try:
        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (64, 64))
        input_data = face_resized.reshape(1, 1, 64, 64).astype(np.float32)
        input_name = emotion_session.get_inputs()[0].name
        pred = emotion_session.run(None, {input_name: input_data})[0][0]
        emotion_idx = np.argmax(pred)
        return emotion_labels[emotion_idx]
    except:
        return "?"


# --- Главный цикл обработки ---
def start_processing_loop():
    global tracked_faces, next_track_id
    cap_thread = FrameCaptureThread(src=0, width=640, height=480)
    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    process_thread = FaceProcessingThread(frame_queue, result_queue)

    cap_thread.start()
    process_thread.start()
    print("🚀 Асинхронное распознавание лиц + эмоции запущено")

    frame_count = 0
    last_detections = []

    try:
        while cap_thread.running:
            ret, frame = cap_thread.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            current_detections = []

            # Получаем детекции
            if not result_queue.empty():
                try:
                    last_detections = result_queue.get_nowait()
                except queue.Empty:
                    pass
            current_detections = last_detections.copy()

            # Отправляем кадр на обработку (раз в 5 кадров)
            if frame_count % 5 == 0 and process_thread.running and frame_queue.empty():
                try:
                    frame_queue.put(frame.copy(), block=False)
                except queue.Full:
                    pass

            # --- Трекинг ---
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

            for idx, det_box in enumerate(current_detections):
                x, y, w, h = det_box
                roi = frame[y:y+h, x:x+w]

                if idx in matched_tracks:
                    track_id = matched_tracks[idx]
                    track = tracked_faces[track_id]
                    name = track["name"]
                    color = track["color"]
                    emotion = track.get("emotion", "")

                    if frame_count % 10 == 0:
                        face_aligned = recognizer.alignCrop(frame, det_box)
                        if face_aligned is not None:
                            emb = recognizer.feature(face_aligned)
                            scores = []
                            for saved_emb in embeddings:
                                score = recognizer.match(emb, saved_emb, cv2.FaceRecognizerSF_FR_COSINE)
                                scores.append(score)

                            best_score = max(scores) if scores else 0.0
                            best_idx = np.argmax(scores) if scores else -1

                            if best_score >= CONFIDENCE_THRESHOLD and best_idx != -1:
                                name = labels[best_idx].split("_", 1)[1]  # например, "ivan"
                                color = (0, 255, 0)  # зелёный
                            else:
                                name = "Unknown"
                                color = (0, 0, 255)  # красный

                        if roi.size > 0:
                            emotion = predict_emotion(roi)

                    tracked_faces[track_id].update({
                        "bbox": det_box,
                        "name": name,
                        "color": color,
                        "emotion": emotion,
                        "age": 0
                    })
                else:
                    tracked_faces[next_track_id] = {
                        "bbox": det_box,
                        "name": "Unknown",
                        "color": (255, 0, 0),
                        "emotion": "",
                        "age": 0
                    }
                    next_track_id += 1

            # Удаление старых
            for tid in list(tracked_faces.keys()):
                if tid not in current_track_ids:
                    tracked_faces[tid]["age"] += 1
                    if tracked_faces[tid]["age"] > MAX_AGE:
                        del tracked_faces[tid]

            # Обновление данных для API
            update_faces_data(tracked_faces)

            time.sleep(0.01)  # чтобы не грузить CPU

    finally:
        cap_thread.stop()
        process_thread.stop()
        print("👋 Цикл обработки остановлен.")


# --- Lifespan: современная замена on_event ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("✅ Сервер запускается... http://127.0.0.1:8000/docs")
    processing_thread = threading.Thread(target=start_processing_loop, daemon=True)
    processing_thread.start()
    yield
    print("👋 Сервер остановлен.")


# --- FastAPI приложение ---
app = FastAPI(
    title="Face Recognition API",
    description="Распознавание лиц и эмоций в реальном времени",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/faces", response_class=JSONResponse)
async def get_faces():
    """
    Возвращает текущий срез распознанных лиц.
    """
    with data_lock:
        return JSONResponse(content=latest_faces_data)


@app.get("/health")
async def health_check():
    """
    Проверка работоспособности сервиса.
    """
    with data_lock:
        return {"status": "healthy", "faces_count": len(latest_faces_data)}


# --- Точка входа для запуска через python (опционально) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("experimental.camera_probe_face_with_btns_and_emotions_async_fast_api:app",
                host="127.0.0.1", port=8000, reload=True)
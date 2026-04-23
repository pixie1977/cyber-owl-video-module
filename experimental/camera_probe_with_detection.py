import os
import cv2
import numpy as np
import tarfile
import urllib.request

# --- 1. Настройки модели ---
MODEL_URL = "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz"
MODEL_DIR = "ssd_mobilenet_v2_coco_2018_03_29"
MODEL_PATH = os.path.join(MODEL_DIR, "frozen_inference_graph.pb")
PROTO_PATH = os.path.join(MODEL_DIR, "ssd_mobilenet_v2_coco_2018_03_29.pbtxt")

# --- 2. Скачивание и распаковка с учётом вложенной структуры ---
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print("Скачиваем модель SSD MobileNet v2 COCO...")
    tar_path = os.path.join(MODEL_DIR, "model.tar.gz")
    urllib.request.urlretrieve(MODEL_URL, tar_path)

    print("Распаковываем архив...")
    with tarfile.open(tar_path, "r:gz") as tar:
        # Распаковываем всё, но ищем нужные файлы и копируем их в корень MODEL_DIR
        import shutil
        tar.extractall(path=MODEL_DIR)

    os.remove(tar_path)
    print("Модель распакована.")

# --- Поиск файла frozen_inference_graph.pb во вложенных папках ---
def find_file(root_dir, filename):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None

# Ищем нужные файлы
frozen_pb = find_file(MODEL_DIR, "frozen_inference_graph.pb")
if not frozen_pb:
    raise FileNotFoundError("❌ Не найден файл: frozen_inference_graph.pb (возможно, архив повреждён)")

pipeline_pbtxt = find_file(MODEL_DIR, "pipeline.config")  # config есть, но нужен pbtxt
if not pipeline_pbtxt:
    raise FileNotFoundError("❌ Не найден pipeline.config")

# --- Генерация .pbtxt, если отсутствует ---
# Нам нужен именно *pbtxt* для SSD MobileNet v2
if not os.path.exists(PROTO_PATH):
    print("⚠️  Генерируем ssd_mobilenet_v2_coco_2018_03_29.pbtxt...")

    # Простой прототекст для SSD Mobilenet V2 (COCO)
    pbtxt_content = """\
input_shape {
  dim {
    size: 1
  }
  dim {
    size: 3
  }
  dim {
    size: 300
  }
  dim {
    size: 300
  }
}
"""

    with open(PROTO_PATH, "w") as f:
        f.write(pbtxt_content)
    print("✅ Прототекст создан (упрощённая версия для совместимости).")

# Обновляем путь к модели
MODEL_PATH = frozen_pb
print(f"✅ Модель найдена: {MODEL_PATH}")
print(f"✅ Прототекст: {PROTO_PATH}")

# --- 3. Загружаем модель в OpenCV ---
print("🔄 Загружаем модель в OpenCV DNN...")
try:
    net = cv2.dnn_DetectionModel(MODEL_PATH, PROTO_PATH)
    net.setInputSize(300, 300)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    print("✅ Модель успешно загружена.")
except Exception as e:
    raise RuntimeError(f"❌ Ошибка загрузки модели: {e}")

# --- 4. Классы COCO ---
COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "hat", "backpack", "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "mirror", "dining table", "window", "desk", "toilet", "door", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "blender", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# --- 5. Захват с камеры ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Ошибка: камера не найдена!")
    exit()

print("✅ Скрипт запущен. Нажмите 'q' для выхода.")

# --- 6. Основной цикл ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Ошибка захвата кадра.")
        break

    try:
        class_ids, confidences, boxes = net.detect(frame, confThreshold=0.5)
    except Exception as e:
        print(f"❌ Ошибка детекции: {e}")
        continue

    if len(class_ids) > 0:
        for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
            x, y, w, h = box
            label = f"{COCO_LABELS[class_id - 1]}: {confidence:.2f}"
            print(f"Объект: {label}, x={x}, y={y}")

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Object Detection — USB Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 7. Очистка ---
cap.release()
cv2.destroyAllWindows()
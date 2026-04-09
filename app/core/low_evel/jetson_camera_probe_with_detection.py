import jetson_inference
import jetson_utils

# 1. Загружаем модель детектора (по умолчанию ищет 91 класс: люди, машины, собаки и т.д.)
# 'ssd-mobilenet-v2' — быстрая и легкая модель
net = jetson_inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# 2. Настраиваем источник (CSI камера)
camera = jetson_utils.videoSource("csi://0")

# 3. Настраиваем вывод (WebRTC стриминг для просмотра в браузере)
display = jetson_utils.videoOutput("webrtc://@:8554/my_stream")

print("Скрипт запущен. Откройте http://<IP_JETSON>:8554 в браузере")

while display.IsStreaming():
    # Захват кадра
    img = camera.Capture()

    if img is None:
        continue

    # Обнаружение объектов
    detections = net.Detect(img)

    # Вывод координат в консоль (опционально)
    for detection in detections:
        print(f"Объект ID {detection.ClassID}: x={detection.Left:.1f}, y={detection.Top:.1f}")

    # Рендеринг (отправка кадра в стрим)
    display.Render(img)

    # Обновление заголовка с FPS
    display.SetStatus("DetectNet | Network {:.0f} FPS".format(net.GetNetworkFPS()))

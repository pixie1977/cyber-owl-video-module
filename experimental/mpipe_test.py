try:
    import mediapipe as mp
    print("✅ Успешно импортирован mediapipe")
    print("Версия:", mp.__version__)

    # Проверяем solutions
    if hasattr(mp, 'solutions'):
        print("✅ mp.solutions ДОСТУПЕН")
        if hasattr(mp.solutions, 'face_detection'):
            print("✅ mp.solutions.face_detection работает")
        else:
            print("❌ mp.solutions НЕ содержит face_detection")
    else:
        print("❌ mediapipe НЕ имеет атрибута 'solutions'")

except ImportError as e:
    print("❌ Не удалось импортировать mediapipe:", e)

except Exception as e:
    print("❌ Ошибка:", e)
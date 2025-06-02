import os
from models import MobileNetClassifier  # nếu bạn đã tách ra file riêng
# from your_django_app.models import MobileNetClassifier  # nếu vẫn để trong models.py

# Cấu hình
TRAIN_DIR = 'data/train'  # Thư mục chứa dữ liệu huấn luyện
MODEL_PATH = 'skin_disease_mobilenet_model.keras'
BATCH_SIZE = 8
EPOCHS = 25
NUM_CLASSES = 5

def main():
    # Khởi tạo mô hình
    classifier = MobileNetClassifier(input_shape=(224, 224, 3), num_classes=NUM_CLASSES)

    # Huấn luyện
    history = classifier.train(
        train_dir=TRAIN_DIR,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )

    # Lưu mô hình
    classifier.save_model(MODEL_PATH)
    print(f"✅ Mô hình đã được lưu tại: {MODEL_PATH}")

if __name__ == '__main__':
    main()

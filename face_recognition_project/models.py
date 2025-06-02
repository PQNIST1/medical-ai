from django.db import models

from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras import layers, models
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.applications import MobileNetV2
from keras._tf_keras.keras.callbacks import LearningRateScheduler
import numpy as np
from scipy.special import softmax
from scipy.stats import entropy


class MobileNetClassifier:
    def __init__(self, input_shape=(224, 224, 3), num_classes=5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        # Sử dụng MobileNetV2 với trọng số từ ImageNet
        base_model = MobileNetV2(input_shape=self.input_shape, include_top=False, weights='imagenet')
    
        # Mở lại các lớp của base_model để huấn luyện
        base_model.trainable = True
    
        # Chỉ huấn luyện các lớp sau cùng của base_model
        fine_tune_at = 100  # Chỉ fine-tune các lớp sau lớp thứ 100
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
    
        # Xây dựng mô hình với các lớp tùy chỉnh ở đầu ra
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')  # Lớp đầu ra với số lớp tùy chỉnh
        ])
    
        # Compile mô hình với Adam optimizer và loss function sparse categorical crossentropy
        model.compile(optimizer=Adam(learning_rate=0.0001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, train_dir, batch_size, epochs):
        # Tạo generator cho dữ liệu huấn luyện
        train_datagen = ImageDataGenerator(rescale=1./255)
       
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='sparse'
        )
        
        # # Huấn luyện mô hình
        # history = self.model.fit(train_generator, epochs=epochs)
        lr_schedule = LearningRateScheduler(lambda epoch: 1e-3 * 0.9**epoch)
        # Huấn luyện mô hình với callback
        history = self.model.fit(train_generator, epochs=epochs, callbacks=[lr_schedule])
        avg_train_accuracy = sum(history.history['accuracy']) / len(history.history['accuracy'])
        print(f"Average Training Accuracy: {avg_train_accuracy * 100:.2f}%")
        return history

    def evaluate(self, test_dir, batch_size=8):
        # Tạo generator cho dữ liệu kiểm tra
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='sparse'
        )

        # Đánh giá mô hình và trả về loss, accuracy
        loss, accuracy = self.model.evaluate(test_generator)
        return loss, accuracy

    def save_model(self, model_path):
        # Lưu mô hình vào đường dẫn chỉ định
        self.model.save(model_path)

    def load_model(self, model_path):
        # Tải mô hình từ file đã lưu
        self.model = models.load_model(model_path)

    def predict(self, image_array, threshold=0.6, entropy_threshold=1.2, temperature=5.0):
        try:
            # Đảm bảo ảnh có shape đúng: (1, H, W, 3)
            if image_array.ndim == 3:
                image_array = np.expand_dims(image_array, axis=0)
            image_array = image_array / 255.0  # Chuẩn hóa

            # Lấy xác suất từ model (đã softmax sẵn)
            probabilities = self.model.predict(image_array)[0]

            # Approximate logits từ xác suất
            logits = np.log(probabilities + 1e-12)

            # Temperature scaling
            scaled_logits = logits / temperature
            scaled_probabilities = softmax(scaled_logits)

            predicted_class = np.argmax(scaled_probabilities)
            max_prob = scaled_probabilities[predicted_class]
            ent = entropy(scaled_probabilities)

            # OOD detection bằng entropy và threshold
            if max_prob < threshold or ent > entropy_threshold:
                print(f'Warning: Low confidence or high uncertainty (max_prob={max_prob:.2f}, entropy={ent:.2f}), image might be unrelated.')
                return None, scaled_probabilities

            return predicted_class, scaled_probabilities

        except Exception as e:
            print(f'Error during prediction from array: {e}')
            return None, None


    def get_class_name(self, class_index):
        class_names = ['Acne', 'Actinic Keratosis', 'Basal Cell Carcinoma', 'Eczema', 'Rosacea']
        if class_index is None or class_index < 0 or class_index >= len(class_names):
            return "Unknown"
        return class_names[class_index]


from django.db import models

from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras import layers, models
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.applications import MobileNetV2
import numpy as np
import os

class MobileNetClassifier:
    def __init__(self, input_shape=(224, 224, 3), num_classes=5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        base_model = MobileNetV2(input_shape=self.input_shape, include_top=False, weights='imagenet')
        base_model.trainable = False

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_dir, batch_size=32, epochs=10):
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='sparse'
        )
        history = self.model.fit(train_generator, epochs=epochs)
        return history

    def evaluate(self, test_dir, batch_size=32):
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='sparse'
        )
        return self.model.evaluate(test_generator)

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        self.model = models.load_model(model_path)

    def predict(self, image_array):
        predictions = self.model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=-1)[0]
        probabilities = predictions[0]
        return predicted_class, probabilities

    def get_class_name(self, class_index):
        class_names = ['Acne', 'Actinic Keratosis', 'Basal Cell Carcinoma', 'Eczema', 'Rosacea']
        return class_names[class_index]


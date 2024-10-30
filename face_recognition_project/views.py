from django.conf import settings
from django.core.files.storage import default_storage
from rest_framework.decorators import api_view
from rest_framework.response import Response
from PIL import Image
import numpy as np
from .models import MobileNetClassifier
import os

classifier = MobileNetClassifier(num_classes=5)
classifier.load_model(os.path.join(settings.BASE_DIR, 'face_recognition_project/skin_disease_mobilenet_model.keras'))

@api_view(['POST'])
def classify_image(request):
    if request.method == 'POST' and 'image' in request.FILES:
        image_file = request.FILES['image']
        image_path = default_storage.save('uploads/' + image_file.name, image_file)
        full_image_path = os.path.join(settings.MEDIA_ROOT, image_path)

        # Preprocess the image
        img = Image.open(full_image_path).resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict the class
        predicted_class, probabilities = classifier.predict(img_array)
        class_name = classifier.get_class_name(predicted_class)

        # Return JSON response
        return Response({
            'class_name': class_name,
            'probabilities': probabilities.tolist(),  # convert numpy array to list for JSON serialization
            'image_url': image_path,
        }, status=200)

    return Response({'error': 'Invalid request or no image provided'}, status=400)

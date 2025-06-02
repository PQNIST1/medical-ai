from django.conf import settings
from django.core.files.storage import default_storage
from rest_framework.decorators import api_view
from rest_framework.response import Response
from PIL import Image
import numpy as np
from .models import MobileNetClassifier
import os
import requests
from io import BytesIO
classifier = MobileNetClassifier(num_classes=5)
classifier.load_model(os.path.join(settings.BASE_DIR, 'face_recognition_project/skin_disease_mobilenet_model.keras'))

SKIN_DISEASE_INFO = {
    'Unknown': {
        'description': 'Không rõ thông tin bệnh.',
        'treatment': 'Không có khuyến nghị.',
        'suggested_meds': [],
        'department': 'Không xác định'
    },
    'Acne': {
        'description': 'Mụn trứng cá là bệnh da phổ biến gây ra bởi lỗ chân lông bị tắc nghẽn bởi dầu và tế bào chết.',
        'treatment': 'Giữ da sạch, dùng thuốc bôi chứa benzoyl peroxide hoặc retinoid. Trường hợp nặng có thể dùng kháng sinh hoặc isotretinoin.',
        'suggested_meds': ['Benzoyl peroxide', 'Adapalene', 'Doxycycline'],
        'department': 'Da liễu'
    },
    'Actinic Keratosis': {
        'description': 'Tổn thương da do tiếp xúc lâu dài với ánh nắng, có thể dẫn đến ung thư da nếu không điều trị.',
        'treatment': 'Dùng kem bôi fluorouracil hoặc imiquimod, liệu pháp lạnh (cryotherapy).',
        'suggested_meds': ['Fluorouracil cream', 'Imiquimod'],
        'department': 'Da liễu'
    },
    'Basal Cell Carcinoma': {
        'description': 'Một loại ung thư da thường gặp nhưng tiến triển chậm và hiếm khi di căn.',
        'treatment': 'Phẫu thuật, liệu pháp laser, hoặc kem bôi điều trị ung thư da.',
        'suggested_meds': ['Imiquimod', '5-fluorouracil (5-FU)'],
        'department': 'Ung bướu, Da liễu'
    },
    'Eczema': {
        'description': 'Viêm da gây ngứa, đỏ, bong tróc. Thường do cơ địa dị ứng hoặc kích ứng.',
        'treatment': 'Dưỡng ẩm, tránh yếu tố kích thích, dùng corticosteroid tại chỗ.',
        'suggested_meds': ['Hydrocortisone', 'Tacrolimus'],
        'department': 'Da liễu'
    },
    'Rosacea': {
        'description': 'Bệnh da mãn tính gây đỏ mặt, nổi mụn, thường xuất hiện ở người lớn.',
        'treatment': 'Tránh tác nhân kích ứng, dùng thuốc bôi metronidazole hoặc uống doxycycline.',
        'suggested_meds': ['Metronidazole gel', 'Doxycycline'],
        'department': 'Da liễu'
    }
}



@api_view(['POST'])
def classify_image(request):
    if request.method == 'POST' and 'image' in request.FILES:
        image_file = request.FILES['image']
        image_path = default_storage.save('uploads/' + image_file.name, image_file)
        full_image_path = os.path.join(settings.MEDIA_ROOT, image_path)
        try:
            img = Image.open(full_image_path).convert('RGB').resize((224, 224))
        except Exception as e:
            return Response({'error': f'Không thể xử lý ảnh: {str(e)}'}, status=400)
        img_array = np.array(img)
        predicted_class, probabilities = classifier.predict(img_array)
        if predicted_class is None:
            class_name = "Unknown"
        else:
            class_name = classifier.get_class_name(predicted_class)
        disease_info = SKIN_DISEASE_INFO.get(class_name)
        if os.path.exists(full_image_path):
            os.remove(full_image_path)
        return Response({
            'class_name': class_name,
            'probabilities': probabilities.tolist(),
            'image_url': image_path,
            'description': disease_info['description'],
            'treatment': disease_info['treatment'],
            'suggested_meds': disease_info['suggested_meds'],
            'department': disease_info['department']
        }, status=200)
    return Response({'error': 'Invalid request or no image provided'}, status=400)

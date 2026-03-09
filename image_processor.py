import joblib
import numpy as np
from PIL import Image
import io
from skimage.transform import resize
import os


class ImageProcessor:
    def __init__(self, model_path="models/dog_breed_knn.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"KNN model file not found at {model_path}")
        self.knn = joblib.load(model_path)
        self.breeds = [
            'n02106166-Border_collie',
            'n02108915-French_bulldog',
            'n02085620-Chihuahua',
            'n02110185-Siberian_husky',
            'n02099601-golden_retriever'
        ]

    def preprocess_image(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_array = np.array(image)
            img_resized = resize(img_array, (128, 128), anti_aliasing=True)
            features = img_resized.flatten()
            return features.reshape(1, -1)
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {str(e)}")

    def classify_image(self, image_bytes):
        features = self.preprocess_image(image_bytes)
        pred_label = self.knn.predict(features)[0]
        confidence = self.knn.predict_proba(features)[0].max()

        return {
            "breed": pred_label,
            "confidence": float(confidence)
        }
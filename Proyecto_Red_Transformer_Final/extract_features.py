# extract_features.py
from transformers import AutoFeatureExtractor, AutoModel
import torch
import numpy as np
import cv2
import os

# Cargar el extractor de características y el modelo
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/dino-vitb16")
model = AutoModel.from_pretrained("facebook/dino-vitb16")

# Función para extraer características de una imagen
def extract_features(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = feature_extractor(images=image_rgb, return_tensors="pt")
    with torch.no_grad():
        features = model(**inputs).last_hidden_state
    return features.squeeze().numpy()

# Directorio con las imágenes (frames extraídos del video)
frames_folder = 'C:\\Users\\edgar\\Desktop\\Inteligencia Artificial\\IA\\prueba\\frames_folder'
feature_folder = 'C:\\Users\\edgar\\Desktop\\Inteligencia Artificial\\IA\\prueba\\feature_folder'
os.makedirs(feature_folder, exist_ok=True)

# Recorrer todas las subcarpetas y procesar las imágenes
for root, dirs, files in os.walk(frames_folder):
    for frame_name in files:
        frame_path = os.path.join(root, frame_name)
        features = extract_features(frame_path)
        
        # Crear una ruta similar en feature_folder
        relative_path = os.path.relpath(frame_path, frames_folder)
        feature_path = os.path.join(feature_folder, f'{relative_path}.npy')
        
        # Crear los directorios necesarios si no existen
        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        
        # Guardar las características
        np.save(feature_path, features)

print(f"Características guardadas en: {feature_folder}")
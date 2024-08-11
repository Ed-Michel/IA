import os
import torch
import numpy as np
from train_classifier import SimpleClassifier 

# Parámetros
input_size = 151296  # Debe ser el mismo que en el entrenamiento
num_classes = 74  # Debe ser el mismo que en el entrenamiento

# Cargar el modelo entrenado
parts_folder = 'C:\\Users\\edgar\\Desktop\\Inteligencia Artificial\\IA\\prueba'
model = SimpleClassifier(input_size, num_classes)
model.load_state_dict(torch.load(os.path.join(parts_folder, 'simple_classifier.pth')))
model.eval()

# Función para cargar características de una imagen
def load_features(image_features_path):
    loaded = np.load(image_features_path)
    return loaded['train_data'], loaded['train_labels']

# Función para identificar persona en una imagen
def identify_person(features):
    # Asegurarse de que las características tienen la forma correcta
    features = features.reshape(1, -1)
    features_tensor = torch.tensor(features, dtype=torch.float32)
    outputs = model(features_tensor)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Identificar persona en nuevos frames
new_frame_features_paths = [
    os.path.join(parts_folder, 'train_data_part_0.npz'),
    os.path.join(parts_folder, 'train_data_part_1.npz'),
    os.path.join(parts_folder, 'train_data_part_2.npz'),
    os.path.join(parts_folder, 'train_data_part_3.npz'),
    os.path.join(parts_folder, 'train_data_part_4.npz'),
    os.path.join(parts_folder, 'train_data_part_5.npz'),
    os.path.join(parts_folder, 'train_data_part_6.npz'),
    os.path.join(parts_folder, 'train_data_part_7.npz'),
    os.path.join(parts_folder, 'train_data_part_8.npz'),
    os.path.join(parts_folder, 'train_data_part_9.npz'),
]

for new_frame_features_path in new_frame_features_paths:
    try:
        features, labels = load_features(new_frame_features_path)
        for i, feature in enumerate(features):
            person_id = identify_person(feature)
            print(f'Identified Person ID in {new_frame_features_path}, sample {i}: {person_id}')
    except KeyError:
        print(f'Error: "train_data" not found in {new_frame_features_path}')
    except FileNotFoundError:
        print(f'Error: File not found {new_frame_features_path}')
    except RuntimeError as e:
        print(f'Runtime error: {e} in file {new_frame_features_path}')
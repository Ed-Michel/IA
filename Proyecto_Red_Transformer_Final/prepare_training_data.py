import os
import numpy as np

# Carpeta donde se guardaron las características
feature_folder = 'C:\\Users\\edgar\\Desktop\\Inteligencia Artificial\\IA\\prueba\\feature_folder'

# Lista para almacenar características y etiquetas
all_features = []
all_labels = []

# Recorrer todas las subcarpetas y archivos de características
for root, dirs, files in os.walk(feature_folder):
    for feature_file in files:
        if feature_file.endswith('.npy'):
            # Cargar las características
            feature_path = os.path.join(root, feature_file)
            features = np.load(feature_path)
            
            # Obtener la etiqueta (nombre de la subcarpeta)
            person_id = os.path.basename(root)
            
            # Añadir a las listas
            all_features.append(features)
            all_labels.append(person_id)

# Convertir listas a matrices numpy
train_data = np.array(all_features)
train_labels = np.array(all_labels)

# Verificar las dimensiones y tipos de datos antes de guardar
print(f"train_data shape: {train_data.shape}, dtype: {train_data.dtype}")
print(f"train_labels shape: {train_labels.shape}, dtype: {train_labels.dtype}")

# Definir la cantidad de partes para dividir los datos
num_parts = 10
data_split = np.array_split(train_data, num_parts)
labels_split = np.array_split(train_labels, num_parts)

# Guardar cada parte por separado
for i, (data_part, labels_part) in enumerate(zip(data_split, labels_split)):
    np.savez(f'C:\\Users\\edgar\\Desktop\\Inteligencia Artificial\\IA\\prueba\\train_data_part_{i}.npz', train_data=data_part, train_labels=labels_part)

print("Datos de entrenamiento guardados en partes.")

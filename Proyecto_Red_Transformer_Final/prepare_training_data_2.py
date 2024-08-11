import numpy as np

# Definir la cantidad de partes que se guardaron
num_parts = 10
train_data_parts = []
train_labels_parts = []

# Cargar cada parte y añadir a las listas
for i in range(num_parts):
    data_path = f'C:\\Users\\edgar\\Desktop\\Inteligencia Artificial\\IA\\prueba\\train_data_part_{i}.npz'
    loaded = np.load(data_path)
    train_data_parts.append(loaded['train_data'])
    train_labels_parts.append(loaded['train_labels'])

# Concatenar todas las partes
train_data = np.concatenate(train_data_parts)
train_labels = np.concatenate(train_labels_parts)

# Verificar las dimensiones y tipos de datos cargados
print(f"loaded_train_data shape: {train_data.shape}, dtype: {train_data.dtype}")
print(f"loaded_train_labels shape: {train_labels.shape}, dtype: {train_labels.dtype}")
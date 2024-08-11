import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import pickle

# Definir la clase del modelo de clasificación simple
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)

# Definir la función para cargar los datos de entrenamiento desde partes
def load_training_data(parts_folder, num_parts):
    train_data_parts = []
    train_labels_parts = []

    for i in range(num_parts):
        data_path = os.path.join(parts_folder, f'train_data_part_{i}.npz')
        loaded = np.load(data_path)
        train_data_parts.append(loaded['train_data'])
        train_labels_parts.append(loaded['train_labels'])

    train_data = np.concatenate(train_data_parts)
    train_labels = np.concatenate(train_labels_parts)

    return train_data, train_labels

# Configuración de las rutas
parts_folder = 'C:\\Users\\edgar\\Desktop\\Inteligencia Artificial\\IA\\prueba'
num_parts = 10

# Cargar los datos de entrenamiento
train_data, train_labels = load_training_data(parts_folder, num_parts)

# Codificar las etiquetas como números enteros
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)

# Guardar el LabelEncoder
with open(os.path.join(parts_folder, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(label_encoder, f)

# Convertir a tensores de PyTorch
train_data = torch.tensor(train_data, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)

# Crear y entrenar el modelo
input_size = train_data.shape[1] * train_data.shape[2]  # Ajuste de input_size
num_classes = len(set(train_labels.numpy()))
model = SimpleClassifier(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento del modelo
for epoch in range(10):
    optimizer.zero_grad()
    inputs = train_data.view(train_data.size(0), -1)  # Ajustar la vista del tensor
    print(f'Size of input: {inputs.shape}')
    print(f'Size of weight matrix: {model.fc.weight.shape}')
    outputs = model(inputs)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Guardar el modelo entrenado
torch.save(model.state_dict(), os.path.join(parts_folder, 'simple_classifier.pth'))
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Definir la carpeta donde se guardan los archivos
parts_folder = 'C:\\Users\\edgar\\Desktop\\Inteligencia Artificial\\IA\\prueba'

# Cargar el LabelEncoder guardado
with open(os.path.join(parts_folder, 'label_encoder.pkl'), 'rb') as file:
    label_encoder = pickle.load(file)

# Decodificar el person_id
person_id = 4  # Cambia este ID según sea necesario
person_name = label_encoder.inverse_transform([person_id])[0]
print(f'The person identified with ID {person_id} is: {person_name}')
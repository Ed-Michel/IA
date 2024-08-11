import cv2
import os
from PIL import Image
import numpy as np

# Función para extraer frames de un video
def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        frame_path = os.path.join(output_folder, f"frame{count:04d}.jpg")
        cv2.imwrite(frame_path, image)
        success, image = vidcap.read()
        count += 1
    print(f"Extraídos {count} frames de {video_path}")

# Función para preprocesar los frames
def preprocess_frame(frame_path, size=(224, 224)):
    image = Image.open(frame_path)
    image = image.resize(size)
    image = np.array(image) / 255.0  # Normalizar a [0, 1]
    return image

# Extraer y preprocesar frames de todos los videos
def process_videos(video_paths, output_folder):
    frame_files = []
    labels = []
    for video_path in video_paths:
        # Obtener el nombre de la carpeta del video
        person_name = os.path.basename(os.path.dirname(video_path))
        person_folder = os.path.join(output_folder, person_name)
        extract_frames(video_path, person_folder)
        frames = []
        for frame_file in sorted(os.listdir(person_folder)):
            frame_path = os.path.join(person_folder, frame_file)
            try:
                frame = preprocess_frame(frame_path)
                frames.append(frame)
            except Exception as e:
                print(f"Error procesando {frame_path}: {e}")
        if frames:
            frames_array = np.array(frames)
            frame_file_path = os.path.join(output_folder, f"{person_name}_frames.npy")
            np.save(frame_file_path, frames_array)
            frame_files.append(frame_file_path)
            labels.append(person_name)
        else:
            print(f"No se encontraron frames en {person_folder}")
    return frame_files, np.array(labels)

# Rutas de los videos y carpeta de salida
video_paths = [
    'Proyecto_Red_Transformer/video-outputs/admin/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/admin (2)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/admin (3)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/admin (4)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/admin (5)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/alan/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/alan (2)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/alan (3)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/alan (4)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/alan (5)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/ame/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/ame (2)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/ame (3)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/ame (4)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/ame (5)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/AndresMendoza1/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/AndresMendoza2/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/AndresMendoza3/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/AndresMendoza4/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/AndresMendoza5/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/Karla(1)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/Karla(2)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/Karla(3)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/LopezLopez(1)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/LopezLopez(2)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/LopezLopez(3)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/LopezLopez(4)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/LopezLopez(5)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/Michel(1)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/Michel(2)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/Michel(3)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/Michel(4)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/Michel(5)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/moki/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/moki (2)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/moki (3)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/moki (4)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/moki (5)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/Noe(1)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/Noe(2)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/Noe(3)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/Noe(4)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/Noe(5)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/PedroSalvador1/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/PedroSalvador2/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/PedroSalvador3/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/PedroSalvador4/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/PedroSalvador5/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/RafaAlberto(1)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/RafaAlberto(2)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/RafaAlberto(3)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/RafaAlberto(4)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/RafaAlberto(5)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/Ramses(1)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/Ramses(2)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/Ramses(3)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/Ramses(4)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/Ramses(5)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/sebastian1/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/sebastian2/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/sebastian3/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/tellez1/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/tellez2/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/tellez3/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/Ximena1/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/Ximena2/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/Ximena3/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/Yahir (1)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/Yahir (2)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/Yahir (3)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/Yuliana(1)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/Yuliana(2)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/Yuliana(3)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/Yuliana(4)/mask.avi',
    'Proyecto_Red_Transformer/video-outputs/Yuliana(5)/mask.avi'
]
output_folder = 'Proyecto_Red_Transformer/frames'

# Ejecutar el procesamiento de videos
frame_files, labels = process_videos(video_paths, output_folder)
print(f"Frames procesados: {len(frame_files)} videos, Labels: {labels.shape}")

# Guardar rutas de archivos de frames y etiquetas
np.save(os.path.join(output_folder, 'frame_files.npy'), frame_files)
np.save(os.path.join(output_folder, 'labels.npy'), labels)
import os
from pathlib import Path
import pandas as pd

# Ruta de tu dataset de audios
dataset_path = Path("../../dataset/")

# Lista de emociones
emotions = [folder for folder in dataset_path.iterdir() if folder.is_dir()]

# Lista para almacenar los datos (ruta del archivo y emoción)
data = []

# Recorrer todas las carpetas de emociones
for emotion_folder in emotions:
    emotion = emotion_folder.name
    # Recorrer todos los archivos MP3 en la carpeta de la emoción
    for audio_file in emotion_folder.glob("*.mp3"):
        # Agregar la ruta del archivo y la emoción
        data.append([str(audio_file), emotion])

# Crear un DataFrame con los datos
df = pd.DataFrame(data, columns=["File Path", "Emotion"])

# Guardar el DataFrame en un archivo CSV
df.to_csv("audio_emotions.csv", index=False)

print("Archivo CSV creado con éxito.")

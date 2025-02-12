import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Función para extraer MFCCs de un archivo de audio
def extract_features(file_path):
    print(f"Extrayendo características del archivo: {file_path}")
    # Cargar el audio
    y, sr = librosa.load(file_path, sr=None)
    
    # Extraer MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    
    # Promedio de los coeficientes MFCC a través del tiempo
    return np.mean(mfcc.T, axis=0)

# Cargar el archivo CSV
print("Cargando el archivo CSV...")
df = pd.read_csv("audio_emotions.csv")

# Lista para almacenar las características y etiquetas
features = []
labels = []

# Recorrer los archivos y extraer las características
print("Recorriendo las filas del CSV...")
for index, row in df.iterrows():
    print(f"Procesando archivo {index + 1}/{len(df)}: {row['File Path']}")
    
    try:
        # Extraer características del audio
        feature = extract_features(row["File Path"])
        features.append(feature)
        
        # Etiqueta de la emoción
        labels.append(row["Emotion"])
    except Exception as e:
        print(f"Error al procesar el archivo {row['File Path']}: {e}")

# Convertir a arrays de numpy
X = np.array(features)
y = np.array(labels)

print(f"Características extraídas: {len(features)}")
print(f"Etiquetas extraídas: {len(labels)}")

# Codificar las etiquetas (emociones) a números
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Dividir el dataset en entrenamiento y prueba
print("Dividiendo el dataset en entrenamiento y prueba...")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Entrenar el modelo
print("Entrenando el modelo RandomForest...")
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluar el modelo
print("Evaluando el modelo...")
y_pred = model.predict(X_test)

# Mostrar resultados
print("Resultados de la evaluación:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

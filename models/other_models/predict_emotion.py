import sys
import librosa
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Función para extraer MFCCs de un archivo de audio
def extract_features(file_path):
    print(f"Extrayendo características del archivo: {file_path}")
    # Cargar el audio
    y, sr = librosa.load(file_path, sr=None)
    
    # Extraer MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    
    # Promedio de los coeficientes MFCC a través del tiempo
    return np.mean(mfcc.T, axis=0)

def load_model(model_path):
    # Cargar el modelo desde el archivo .pkl
    model = joblib.load(model_path)
    return model

def load_label_encoder(label_encoder_path):
    # Cargar las clases del LabelEncoder desde el archivo .npy
    label_classes = np.load(label_encoder_path)
    
    # Crear un LabelEncoder con las clases cargadas
    label_encoder = LabelEncoder()
    label_encoder.classes_ = label_classes
    
    return label_encoder

def predict_emotion(model, label_encoder, audio_path):
    # Extraer características del audio
    features = extract_features(audio_path)
    
    # Convertir a array numpy
    features = np.array([features])
    
    # Hacer la predicción con el modelo cargado
    predicted_emotion = model.predict(features)
    
    # Decodificar la predicción usando el LabelEncoder
    predicted_label = label_encoder.inverse_transform(predicted_emotion)
    
    return predicted_label[0]

if __name__ == '__main__':
    # Comprobar si el argumento fue pasado correctamente
    if len(sys.argv) != 2:
        print("Por favor, pasa la ruta al archivo de audio como argumento.")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    # Rutas de los archivos del modelo y LabelEncoder
    model_path = 'emotion_recognition_model.pkl'  # Ajusta el nombre del archivo si es necesario
    label_encoder_path = 'label_encoder_classes.npy'  # Ajusta el nombre del archivo si es necesario
    
    # Cargar el modelo y el LabelEncoder
    model = load_model(model_path)
    label_encoder = load_label_encoder(label_encoder_path)
    
    # Predecir la emoción del audio
    predicted_emotion = predict_emotion(model, label_encoder, audio_path)
    
    # Imprimir la emoción predicha
    print(f"La emoción predicha para el audio {audio_path} es: {predicted_emotion}")

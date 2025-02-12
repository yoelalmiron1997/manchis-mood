import sys
import librosa
import numpy as np
import joblib
import pygame
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog, 
    QVBoxLayout, QProgressBar, QStyle, QHBoxLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from sklearn.preprocessing import LabelEncoder

# Función para extraer características de un archivo de audio
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.mean(mfcc.T, axis=0)

# Cargar el modelo entrenado
def load_model(model_path):
    return joblib.load(model_path)

# Cargar el codificador de etiquetas
def load_label_encoder(label_encoder_path):
    label_classes = np.load(label_encoder_path)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = label_classes
    return label_encoder

# Clase para procesar la predicción en un hilo separado (evita que la interfaz se congele)
class PredictionThread(QThread):
    progress = pyqtSignal(int)
    result = pyqtSignal(str)

    def __init__(self, model, label_encoder, audio_path):
        super().__init__()
        self.model = model
        self.label_encoder = label_encoder
        self.audio_path = audio_path

    def run(self):
        try:
            self.progress.emit(20)  
            features = extract_features(self.audio_path)
            self.progress.emit(60)  
            features = np.array([features])  # Convertir a formato compatible con el modelo
            predicted_emotion = self.model.predict(features)
            self.progress.emit(90)  
            predicted_label = self.label_encoder.inverse_transform(predicted_emotion)
            self.progress.emit(100)  
            self.result.emit(predicted_label[0])
        except Exception as e:
            self.result.emit(f"Error: {e}")

# Clase para la interfaz gráfica con PyQt5
class EmotionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.model = load_model("./train_model/emotion_recognition_model.pkl")
        self.label_encoder = load_label_encoder("./train_model/label_encoder_classes.npy")
        self.audio_path = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Predicción de Emoción en Audio 🎵")
        self.setGeometry(100, 100, 450, 250)

        # Botón para cargar el archivo
        self.btn_load = QPushButton("🎵 Seleccionar Audio", self)
        self.btn_load.clicked.connect(self.load_audio)

        # Botón para reproducir el audio
        self.btn_play = QPushButton(self.style().standardIcon(QStyle.SP_MediaPlay), " Reproducir", self)
        self.btn_play.clicked.connect(self.play_audio)
        self.btn_play.setEnabled(False) 

        # Barra de progreso
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)

        # Etiqueta para mostrar el resultado
        self.result_label = QLabel("Emoción estimada: -", self)
        self.result_label.setAlignment(Qt.AlignCenter)

        # Layout de los botones
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_load)
        button_layout.addWidget(self.btn_play)

        # Layout principal
        layout = QVBoxLayout()
        layout.addLayout(button_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

    def load_audio(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar archivo de audio", "", "Archivos de audio (*.mp3 *.wav)", options=options)

        if file_path:
            self.audio_path = file_path
            self.btn_play.setEnabled(True)  
            self.predict_audio_emotion()

    def play_audio(self):
        if self.audio_path:
            pygame.mixer.init()
            pygame.mixer.music.load(self.audio_path)
            pygame.mixer.music.play()

    def predict_audio_emotion(self):
        self.progress_bar.setValue(10)  # Inicializa la barra de progreso
        self.result_label.setText("Analizando... ⏳")

        # Hilo para ejecutar la predicción sin congelar la interfaz
        self.thread = PredictionThread(self.model, self.label_encoder, self.audio_path)
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.result.connect(self.display_result)
        self.thread.start()

    def display_result(self, predicted_emotion):
        self.result_label.setText(f"Emoción estimada: {predicted_emotion} 🐱")

# Ejecutar la aplicación
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionApp()
    window.show()
    sys.exit(app.exec_())

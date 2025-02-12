import librosa
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

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
df = pd.read_csv("../extract_data/audio_emotions.csv")

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

# Ajustar el modelo con pesos balanceados
print("Entrenando el modelo RandomForest con pesos balanceados...")
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Realizar validación cruzada
cv_scores = cross_val_score(model, X, y_encoded, cv=5)
print(f"Precisión de la validación cruzada: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Ajuste de hiperparámetros con Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(class_weight='balanced', random_state=42), 
                           param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)
print(f"Mejores parámetros encontrados: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

# Evaluar el modelo con el mejor ajuste de hiperparámetros
print("Evaluando el modelo...")
y_pred = best_model.predict(X_test)

# Mostrar resultados
print("Resultados de la evaluación:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Obtener las probabilidades para AUC-ROC
y_prob = best_model.predict_proba(X_test)

# AUC-ROC para cada clase
auc_scores = roc_auc_score(y_test, y_prob, multi_class='ovr')
print(f"AUC-ROC para cada clase: {auc_scores}")

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:")
print(conf_matrix)

# Guardar el modelo entrenado
print("Guardando el modelo entrenado...")
joblib.dump(best_model, 'emotion_recognition_model.pkl')

# Guardar el LabelEncoder (para convertir las predicciones de vuelta a emociones)
print("Guardando el LabelEncoder...")
np.save('label_encoder_classes.npy', label_encoder.classes_)

print("Modelo y LabelEncoder guardados con éxito.")

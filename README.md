# 🐱 Manchis Mood 🎶

Este proyecto permite cargar archivos de audio (.mp3) de gatos para analizar y predecir la emoción expresada en sus sonidos. Utiliza técnicas de procesamiento de audio y aprendizaje automático para clasificar las emociones mas comunes y mostrarlas de manera visual.


## Estructura del proyecto

```yaml
audio_test/
├── WhatsApp-Audio1.mp3                # Audios formato .mp3 para predecir
├── WhatsApp-Audio2.mp3                # Audios formato .mp3 para predecir
│
dataset/                    
│
├── angry/                  # Audios formato .mp3 'angry'
├── defense/                # Audios formato .mp3 'defense'
├── fighting/               # Audios formato .mp3 'fighting'
├── happy/                  # Audios formato .mp3 'happy'
├── mother_call/            # Audios formato .mp3 'mother_call'
├── paining/                # Audios formato .mp3 'paining'
└── warning/                # Audios formato .mp3 'warning'

models/                      # Carpeta que contiene todos los modelos y scripts asociados
│
├── extract_data/            # Script y archivos para cargar y procesar los datos CSV
│   ├── .gitignore           
│   ├── audio_emotions.csv   
│   └── data_set_csv.py      # Script para procesar los datos del CSV
│
├── other_models/            
│
├── train_model/             # Scripts y archivos necesarios para entrenar el modelo
│   ├── .gitignore          
│   ├── emotion_recognition_model.pkl   # Modelo entrenado para reconocimiento de emociones
│   ├── extract_audio_max.py # Script para extraer características de audio
│   └── label_encoder_classes.npy  # Archivo con las clases de etiquetas codificadas
│
└── predict_emotion_app.py   # Script principal 

requirements.txt            
README.md                   

```

## Instalación
```bash
git clone https://github.com/yoelalmiron1997/mi-proyecto-ia-animales.git
pip install -r requirements.txt
```

## 📋 Extracción 
Se extraen la información del `dataset` generando un archivo csv. 

```bash
python data_set_csv.py
```

##### 📌 Estructura de CSV de salida

El archivo de salida contiene la clasificación de los archivos de audios analizados, asociado cada archivo con una emoción detectada.

```bash
<RUTA_DEL_ARCHIVO>,<EMOCIÓN_PREDICHA>
```

```bash
audio_emotions.csv << archivo generado
```

## 📜 Modelo de entrenamiento

El modelo entrenado está basado en un Random `Forest Classifier` y las caracteristicas extraídas de los audios son mediante `MFCC` (Coeficientes Cepstrales en las Frecuencias de Mel).

```text
MFCC Coefficients over Time (Promedio por frame)
--------------------------------------------------------------------------------------------------------------------------
| Time (s) |  MFCC-0  |  MFCC-1  |  MFCC-2  |  MFCC-3  |  MFCC-4  |  MFCC-5  |  MFCC-6  |  MFCC-7  |  MFCC-8  |  MFCC-9  |
--------------------------------------------------------------------------------------------------------------------------
|   t1     |  12.34   |   5.67   |   3.45   |   2.12   |   1.08   |   0.78   |   0.56   |   1.45   |   2.34   |   3.21   |
|   t2     |  11.45   |   5.12   |   3.22   |   1.98   |   0.99   |   0.88   |   0.62   |   1.43   |   2.21   |   3.10   |
|   t3     |  13.12   |   5.87   |   3.67   |   2.13   |   1.05   |   0.79   |   0.53   |   1.47   |   2.40   |   3.23   |
|   t4     |  12.25   |   5.45   |   3.10   |   2.01   |   1.03   |   0.77   |   0.55   |   1.44   |   2.29   |   3.15   |
|   t5     |  12.65   |   5.90   |   3.72   |   2.21   |   1.12   |   0.81   |   0.59   |   1.48   |   2.36   |   3.18   |
--------------------------------------------------------------------------------------------------------------------------
```

Librosa nos permite cargar el archivo de audio y extraer sus características. MFCC son caracteristicas acústicas comúnmente usadas en el procesamiento de señales de audio, especificamente en el campo de reconocimiento de voz y emociones.

### Modelo Random Forest

Es un modelo que posee un conjunto de árboles de decisión, para clasificar las emociones a partir de las características extraídas.

Permite ajustar el modelo por medio de los pesos balanceados en las clases, en caso de algunas emociones son más frecuentes que otras en el dataset.

Se realiza `validación cruzada` para evaluar la estabilidad y la generalización del modelo. Esto implica entrenar y evaluar el modelo en diferente subconjuntos extraidos del dataset para obtener una mejor estimación, mejorando su rendimiento.

Por ultimo se realiza `Hiperparámetros` (Grid Search), esto es usado para optimizar los parámetros del modelo, como el número de árboles, la profundidad máxima de la busqueda y el número mínimo de muestras requeridas para dividir un nodo. 

#### 🔢 Calculos

Se generan los siguientes calculos:
    
- Reporte de clasificación (classification_report) 
- AUC (Área bajo la curva) para indicar el rendimiento.
- Matriz de confusión, muestra como se comporto al predecir las emociones. 


## 👤 Autor

Este proyecto fue desarrollado por **[Yoel Almirón]**. Puedes encontrarme en [GitHub](https://github.com/yoelalmiron1997) or [LinkedIn](https://ar.linkedin.com/in/yoel-almiron). ¡No dudes en contactarme si tienes alguna pregunta o sugerencia! 

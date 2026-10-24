# Segmentación de Filamentos Solares en C++ (LibTorch & OpenCV)

[![C++](https://img.shields.io/badge/C%2B%2B-17-blue)](https://en.cppreference.com/)
[![LibTorch](https://img.shields.io/badge/PyTorch-LibTorch_C%2B%2B-ee4c2c)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)](https://opencv.org/)



Proyecto de visión por computador y Deep Learning desarrollado en C++ para la detección y segmentación semántica de filamentos solares en imágenes H-alpha. En este proyecto implemento una arquitectura U-Net completa desde cero utilizando LibTorch y OpenCV para el procesamiento de imagen. La idea de este proyecto viene del torneo de kaggle https://www.kaggle.com/competitions/filament-segmentation-2026

---

## Stack Tecnológico

* **Lenguaje:** C++17
* **Framework de Deep Learning:** LibTorch
* **Visión por Computador:** OpenCV 4.x
* **Formatos y APIs:** COCO API (`maskApi.h`), `nlohmann/json`
* **Entorno de Compilación:** CMake (soporte en Windows y GCC en Linux/Colab)


<div style="background-color: #eaf2f8; padding: 12px; border-left: 5px solid #2980b9; border-radius: 5px;">

<strong style="color: #2980b9;">Ejecución rápida</strong>

Para una una ejecución rápida del código sin necesidad de descargar nada, puedes utilizar directamente [Google Colab](https://drive.google.com/file/d/1Rah5sGWl8vcX9g1JCVHoKb3rZFunhCJT/view?usp=sharing).

</div>


## Muestra del Modelo

| Imagen Solar Original | Máscara de Predicción |
| :---: | :---: |
| ![Original](https://github.com/857and/Segmentacion-de-Filamentos-Solares-en-C-LibTorch-OpenCV/blob/67cc9191c0c2401295b425221eb7e5eabd3fa90f/imagen_readme/20111114063134Lh.jpeg) | ![Predicción](https://github.com/857and/Segmentacion-de-Filamentos-Solares-en-C-LibTorch-OpenCV/blob/67cc9191c0c2401295b425221eb7e5eabd3fa90f/imagen_readme/pred_20111114063134Lh.jpeg) |


## Estructura del Repositorio

```text
pred/                                    # Predicciones del modelo 
SolarFilamentSegmentation/
   ├── Solar Filament Segmentation.cpp   # Código fuente principal
   ├── maskApi.h / maskApi.c             # API de C para    codificación COCO RLE
   ├── json.hpp                          # Parser JSON header-only
   └── CMakeLists.txt                    # Configuración de compilación
├── README.md                            # Documentación del proyecto
└── analisis.ipynb                       # Archivo de análisis de dataset
```

---

## Arquitectura de la Red (U-Net)

```text
Entrada (1, 512, 512)
   │
   ├──► Encoder: Convoluciones dobles + MaxPool (32 -> 64 -> 128 -> 256)
   │
   ├──► Bottleneck: Conv2D (512)
   │
   ├──► Decoder: ConvTranspose2D + Concatenación (Skip Connections)
   │
   └──► Salida Logits (1, 512, 512) ──► Sigmoid ──► Máscara Binaria
```


## Proceso de Desarrollo y Decisiones Técnicas

### 1. Carga de Datos y Preprocesamiento
* **Parseo COCO JSON:** Para el procesamiento inicial utilicé el parseo de anotaciones en formato COCO JSON debido a que es el estándar en el que se proporcionan las respuestas y segmentaciones para el entrenamiento del dataset.

* **Recorte ROI y Máscara Circular:** Las imágenes solares contienen un alto porcentaje de fondo negro innecesario alrededor del disco solar. He usado una máscara circular combinada con un recorte ROI (1810x1810) para eliminar los detalles fuera de la circunferencia del Sol y centrar el modelo exclusivamente en los filamentos. 


### 2. Entrenamiento y Funciones de Pérdida
* **Combinación de Focal Loss + Dice Loss:** Tras probar distintas configuraciones, la combinación de Focal Loss y Dice Loss fue la que mejor rendimiento me ofreció durante la inferencia. Esto me ha permitido solucionar el fuerte desbalance de clases, ya que los filamentos solares representan un área mínima en comparación con la superficie total del Sol.

### 3. Inferencia

* **Inferencia por Ventana Deslizante:** He aplicado un sistema de particionamiento en parches de 512x512 con solapamiento promediando la probabilidad en las zonas donde se solapan las ventanas para maximizar la precisión de cada pixel individual. _NOTA: Sinceramente no he obtenido mejoras notables a la hora del entranamiento pero yo creo que eso se debe a que necesite más entrenamiento, ya que ahora el modelo dispone de imágenes más diversas y un poco más difíciles de analizar. Por este motivo he decidido dejarlo en el modelo._

### 4. Postprocesado y Exportación con Mask API
* **Codificación RLE:** Para la exportación final de los resultados he utilizado la librería `maskApi.h`. Una vez separados los filamentos mediante `cv::connectedComponentsWithStats`, la API me permite codificar las máscaras binarias directamente a formato Run-Length Encoding (RLE) para la generación del archivo CSV final.


<!-- 888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888 -->

---

## Proceso de Investigación y Desarrollo (Analisis.py)

El objetivo de este archivo complementario es intentar ver posibles aplicaciones de preprocesado y postprocesado a las imágenes. Algunos de los análisis se han aplicado para la inferencia, otros los he descartado.

### Fase de Exploración y Análisis de Datos
Mediante scripts de prueba, he evaluado múltiples enfoques para el tratamiento de las imágenes:
* **Análisis Geométrico y Estadístico:** He usado un script para extraer las métricas de los filamentos (áreas, bounding boxes, diagonales y aspect ratios) mediante funciones de contornos de OpenCV. Esto no vale mucho más que para hacerse una idea de la distribución real de las estructuras en el dataset o intentar encontrar un patrón para el procesamiento de imágenes. Finalmente, los datos obtenidos del análisis de los segmentos permitieron fijar un umbral en el postprocesado (`MIN_PIXELS = 200`) para eliminar falsos positivos y ruido.

* **Preprocesamiento Avanzado y Corrección del Limbo:** He probado técnicas de mejora de contraste adaptativo (CLAHE) y métodos de aplanamiento de fondo como corrección del oscurecimiento del limbo solar mediante desenfoques gaussianos dinámicos calculados al 8% del tamaño de la imagen y normalización Z-score sobre la región central. Estas son un par de técnicas que he visto que se usan a la hora de analizar radiografías para mejorar el análisis de la imagen, ya que quitan desbalances y facilita el análisis de la foto al modelo. Al final, he descartado las técnicas de aplanamiento de limbo (Z-score) y CLAHE para la ejecución debido a que simplemente no había visto una mejora visual notable y he decidido que no merecía la pena su implementación por el tiempo que conllevaba.

<!-- 888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888 -->


## Justificación de la Arquitectura y Decisiones de Diseño

Estas decisiones de diseño e implementación fueron tomadas para superar las limitaciones de hardware del entorno gratuito de **Google Colab** (**16 GB** de RAM).

Para evitar errores de memoria y no saturar el sistema, he organizado los datos de la siguiente manera:

### 1. Recortes a Demanda

* **Funcionamiento:** En lugar de recortar y duplicar miles de parches de 512x512 píxeles en la RAM de antemano, el dataset mantiene las imágenes base en caché y **calcula el recorte de forma individual y aleatoria** justo en el momento en que el `DataLoader` lo solicita.
* **Motivo:** Esto reduce el consumo de la memoria RAM, ya que implicaría precargar combinaciones estáticas de recortes en disco o memoria.

### 2. Lotes Pequeños y Acumulación de Gradientes

Para mantener un entrenamiento estable sin exigir de más a los recursos limitados del entorno, he configurado un tamaño de lote pequeño combinado con acumulación de gradientes:

* **Tamaño del lote (*batch*):** Se procesan grupos pequeños de **4 imágenes** simultáneamente.
* **Frecuencia de actualización de pesos:** Los gradientes se calculan y se van sumando durante exactamente **2 iteraciones** antes de que el optimizador actualice los pesos y limpie los gradientes.
* **Motivo:** Esto significa que el modelo analiza **2 lotes consecutivos de 4 imágenes** (acumulando un total de 8 muestras de procesamiento) antes de realizar un paso de optimización.

## Resumen

Esta estrategia combinada (recortes bajo demanda, lotes pequeños y acumulación de gradientes) permite simular los lotes más grandes sin la necesidad de requerir una gran cantidad de memoria RAM. De esta forma, se evita saturar los recursos y se logra completar el entrenamiento sin que Google Colab cancele el proceso por falta de memoria.
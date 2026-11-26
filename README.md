# Sistema de Detección de Placas y Pico y Placa en Popayán

Sistema de visión computacional que **detecta placas de vehículos particulares en video**, **lee sus caracteres con una CNN entrenada desde cero**, **verifica el pico y placa de Popayán** y responde preguntas en lenguaje natural mediante un asistente conversacional.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completo-brightgreen)

---

## Tabla de Contenidos
- [Descripción General](#descripción-general)
- [Resultados Destacados](#resultados-destacados)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Pipeline de Visión Computacional](#pipeline-de-visión-computacional)
- [CNN de Caracteres](#cnn-de-caracteres)
- [Reglas de Pico y Placa](#reglas-de-pico-y-placa)
- [Asistente Conversacional](#asistente-conversacional)
- [Evolución del Proyecto (v1 → v2)](#evolución-del-proyecto-v1--v2)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Requisitos e Instalación](#requisitos-e-instalación)
- [Guía de Uso](#guía-de-uso)
- [Lecciones Aprendidas y Mejoras Futuras](#lecciones-aprendidas-y-mejoras-futuras)

---

## Descripción General

El proyecto resuelve un problema real de movilidad urbana en Popayán (Colombia) integrando tres bloques tecnológicos:

1. **Visión computacional clásica (OpenCV):** localiza placas amarillas por color y forma, corrige la perspectiva y segmenta los 6 caracteres.
2. **Deep Learning (TensorFlow/Keras):** una CNN propia de 36 clases (`0-9`, `A-Z`) reconoce cada carácter sin usar modelos preentrenados.
3. **Lógica de dominio + LLM:** reglas configurables de pico y placa (rotación, horario, festivos, sanciones) y un asistente que consulta esas reglas a través de *tool calling*.

**Puntos clave del diseño:**

- Los datos de entrenamiento son **sintéticos** (`generador_sintetico.py`): se dibujan placas colombianas y se degradan con perspectiva, desenfoque, ruido, sombras y compresión JPEG, pasando por la **misma segmentación** que se usa en tiempo real.
- La regla de formato `ABC123` (3 letras + 3 números) se aplica como **conocimiento de dominio sobre las probabilidades de la CNN**, corrigiendo confusiones típicas (`O/0`, `I/1`, `B/8`, `S/5`, `Z/2`).
- En video se usa **votación temporal**: la misma placa leída en varios frames seguidos solo se confirma si coincide, evitando cambios por un frame borroso.
- Las reglas de pico y placa **no están escritas en el código**: se leen de `config_popayan.json` y se pueden actualizar si la Alcaldía cambia la rotación.

---

## Resultados Destacados

### CNN de caracteres (36 clases)

| Métrica | Valor |
| :--- | :---: |
| Precisión en prueba (11.340 caracteres) | **96.31%** |
| Pérdida en prueba | `0.2428` |
| Épocas / Tiempo de entrenamiento (CPU) | `19` épocas / `40.9 min` |
| Parámetros | `822.340` |
| Datos (entrenamiento / validación / prueba) | `61.069` / `6.785` / `11.340` |

### Sistema completo (300 escenas sintéticas sobre fondos reales)

| Métrica | v1 | **v2 (final)** |
| :--- | :---: | :---: |
| Detección de placas (IoU ≥ 0.5) | `59.67%` | **94.00%** |
| Segmentación (sobre detectadas) | `68.16%` | **91.84%** |
| Precisión por carácter (CNN) | `96.99%` | **96.07%** |
| Placas completas correctas (sobre segmentadas) | `90.98%` | **93.44%** |
| **Sistema completo (placas correctas / 300)** | `37.00%` | **80.67%** |
| Falsos positivos (51 imágenes sin placa) | `1` | **0** |
| Velocidad | `53.2 FPS` | `16.8 FPS` (`59.6 ms/img`) |

### Comparación con OCR clásico (Tesseract)

Sobre las mismas placas ya segmentadas:

| Motor | Placas completas correctas | Precisión por carácter |
| :--- | :---: | :---: |
| **CNN propia + regla de formato** | **93.44%** | **96.07%** |
| Tesseract (OCR clásico) | `61.39%` | `77.22%` |

> La CNN propia mejora en **+32 puntos** la tasa de acierto del OCR clásico en placas completas, porque fue entrenada específicamente con la tipografía y las degradaciones de las placas colombianas.

---

## Arquitectura del Sistema

```text
┌─────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐
│  Cámara /   │──▶│  OpenCV      │──▶│  CNN propia   │──▶│  Regla de    │
│  Video      │   │  Detección + │   │  36 clases    │   │  formato     │
│  (frames)   │   │ Segmentación │   │  (32×32 gris) │   │  ABC123      │
└─────────────┘   └──────────────┘   └───────────────┘   └──────┬───────┘
                                                               │
                                                    ┌──────────▼─────────┐
                                                    │  pico_placa.py     │
                                                    │  ¿Puede circular?  │
                                                    └──────────┬─────────┘
                                                               │
                                     ┌─────────────────────────▼────────┐
                                     │  app_tiempo_real.py / asistente  │
                                     │  (overlay en video + registro)   │
                                     └──────────────────────────────────┘
```

| Módulo | Responsabilidad |
| :--- | :--- |
| `placas_core.py` | Detección por color HSV + forma, rectificación de perspectiva y segmentación de caracteres (OpenCV). |
| `generador_sintetico.py` | Genera placas sintéticas, degradaciones realistas y escenas de prueba con respuesta conocida. |
| `lector_placas.py` | Pipeline completo imagen → texto, regla de formato, votación temporal y dibujo de resultados. |
| `entrenar_cnn_caracteres.py` | Entrena la CNN de 36 clases desde cero y guarda `modelos/cnn_caracteres.keras`. |
| `pico_placa.py` | Verifica restricciones, dígitos del día, próximos días y explicaciones legibles. |
| `config_popayan.json` | Rotación vigente, horario, sanción y excepciones (editable). |
| `asistente.py` | Agente conversacional con *tool calling* y modo de respaldo sin conexión. |
| `app_tiempo_real.py` | Aplicación de cámara/video en tiempo real con overlay y registro de detecciones. |
| `evaluar_sistema.py` | Evalúa detección, segmentación, lectura, falsos positivos y velocidad. |
| `crear_video_demo.py` | Genera un video de prueba cuando no hay cámara disponible. |

---

## Pipeline de Visión Computacional

Cada frame sigue cinco etapas:

1. **Detección (`detectar_placas`)** — combina dos vías complementarias:
   - *Vía color:* máscara HSV del amarillo colombiano (`H 15-36`), cierre morfológico para unir el fondo que los caracteres "parten" y filtrado por rectangularidad.
   - *Vía bordes:* cuadriláteros convexos de 4 lados sobre el mapa de Canny (detecta el marco negro aunque el vehículo sea amarillo).
   - Cada candidato se verifica por su contenido real (fracción amarilla ≥ 55% y 6%-55% de píxeles oscuros) y se ordena por puntaje, aplicando **supresión de no máximos** (IoU < 0.3).
2. **Rectificación (`rectificar`)** — transformación de perspectiva que deja la placa de frente a `330×160` px (proporción real 33×16 cm), con un pequeño margen para no cortar caracteres.
3. **Binarización y limpieza** — umbral adaptativo gaussiano que tolera sombras y reflejos, y apertura morfológica que borra el marco sin pegarse a los caracteres.
4. **Segmentación (`segmentar_caracteres`)** — componentes conectados filtrados por tamaño (22%-85% de la altura) y posición (franja superior). Si aparecen más de 6 cajas, elige la ventana de 6 más pareada en altura y alineación vertical.
5. **Normalización** — cada carácter se centra en un lienzo cuadrado conservando su proporción (una `I` no se estira) y se lleva a `32×32` en gris.

---

## CNN de Caracteres

Arquitectura estilo VGG con `822.340` parámetros, entrada `32×32×1`:

- 3 bloques convolucionales: `2 × Conv2D 3×3` + `BatchNormalization` + `ReLU` + `MaxPooling` con `32 → 64 → 128` filtros.
- `Flatten (2048)` → `Dense 256` + `BatchNorm` + `ReLU` + `Dropout 0.4` → `Dense 36 (softmax)`.
- **Data augmentation** integrada en el modelo (traslación ±8%, rotación ±11°, zoom 10%), activa solo durante el entrenamiento.
- **Entrenamiento:** `Adam (1e-3)`, `batch=128`, `ReduceLROnPlateau` y `EarlyStopping` monitoreando `val_accuracy`.

**Datos:** 12.000 placas sintéticas de entrenamiento (semilla 1) y 2.000 de prueba (semilla 999, placas nunca vistas), con 691 fallos de segmentación descartados en entrenamiento.

**Confusiones más frecuentes:** `6→2`, `1→2`, `4→2`, `5→2`, `7→2`, `7→4`, `M→N`, `O→0`. La regla de formato elimina gran parte de estos errores en la salida final.

---

## Reglas de Pico y Placa

`pico_placa.py` implementa la lógica de restricción leyendo `config_popayan.json`:

- **Rotación:** lunes `9,0` · martes `1,2` · miércoles `3,4` · jueves `5,6` · viernes `7,8`.
- **Horario:** `07:00` a `20:00`, de lunes a viernes; no aplica sábados, domingos ni festivos (festivos de Colombia vía `holidays`).
- **Formatos válidos:** carro `ABC123` y moto `ABC12D`.
- **Sanción:** infracción C.14 del Código Nacional de Tránsito (multa e inmovilización).

La función `verificar()` devuelve no solo un booleano, sino una **explicación en lenguaje natural** (por ejemplo, si el día no aplica, si la placa está fuera de horario o si tiene restricción), lo que la hace ideal para alimentar al asistente.

---

## Asistente Conversacional

`asistente.py` permite preguntar en lenguaje natural:

- *"¿La placa JMK125 puede salir hoy?"*
- *"¿Qué placas tienen pico y placa el viernes?"*
- *"¿Qué placas detectó la cámara?"*
- *"¿Cuánto es la multa?"*

**Modo generativo:** un LLM (API de Anthropic) interpreta la pregunta y decide qué **herramienta** Python invocar. Las herramientas son funciones reales del proyecto (`verificar_placa`, `digitos_del_dia`, `historial_detecciones`, `reglas_pico_y_placa`), de modo que **el modelo nunca inventa la regla**: la consulta en `pico_placa.py`. Requiere la variable de entorno `ANTHROPIC_API_KEY`.

**Modo de respaldo:** si no hay clave o conexión, un sistema de expresiones regulares responde las mismas preguntas básicas, por lo que el proyecto **funciona sin internet**.

---

## Evolución del Proyecto (v1 → v2)

La comparación entre `metricas_*_v1.json` y `metricas_*.json` documenta una mejora sustancial:

| Aspecto | v1 | v2 | Qué cambió |
| :--- | :---: | :---: | :--- |
| Detección | `59.67%` | `94.00%` | Se añadió la **vía por bordes** (marco negro) y verificación por contenido, no solo por color. |
| Sistema completo | `37.00%` | `80.67%` | Consecuencia directa de la mejor detección y segmentación. |
| Falsos positivos | `1` | `0` | Verificación estricta de color y forma descarta objetos amarillos. |
| Fallos de segmentación (entrenamiento) | `2.198` | `691` | Segmentación más robusta genera más caracteres útiles por placa. |
| Velocidad | `53.2 FPS` | `16.8 FPS` | La detección dual es más costosa, pero sigue por encima de tiempo real (>15 FPS). |

---

## Estructura del Repositorio

```text
proyecto_placas/
├── placas_core.py                 # Detección, rectificación y segmentación (OpenCV)
├── generador_sintetico.py         # Placas sintéticas, degradaciones y escenas de prueba
├── lector_placas.py               # Pipeline completo + regla de formato + votación temporal
├── entrenar_cnn_caracteres.py     # Entrenamiento de la CNN desde cero
├── pico_placa.py                  # Reglas de pico y placa y explicaciones
├── config_popayan.json            # Rotación vigente editable
├── asistente.py                   # Agente conversacional (LLM + modo de respaldo)
├── app_tiempo_real.py             # Aplicación de cámara/video en tiempo real
├── evaluar_sistema.py             # Evaluación cuantitativa del sistema
├── crear_video_demo.py            # Genera video de prueba
├── demo_proyecto.ipynb            # Recorrido paso a paso en Jupyter
├── requirements.txt
├── .gitignore
├── modelos/
│   └── cnn_caracteres.keras       # CNN entrenada (36 clases)
├── fuentes/                       # Tipografías libres para las placas sintéticas
├── fondos/                        # Fotos de vehículos y calles para las pruebas
└── resultados/
    ├── metricas_cnn.json          # Métricas de la CNN
    ├── metricas_sistema.json      # Métricas del sistema completo
    ├── metricas_*_v1.json         # Métricas de la versión anterior
    ├── curvas_cnn_caracteres.png  # Curvas de entrenamiento
    ├── muestras_entrenamiento.png # Ejemplos de caracteres
    ├── figura_pipeline.jpg        # Diagrama del pipeline
    ├── ejemplos_deteccion.jpg     # Detecciones y lecturas de ejemplo
    ├── demo_tiempo_real.jpg       # Captura de la app en tiempo real
    ├── video_demo.mp4             # Video de prueba (sin overlay)
    ├── video_demo_resultado.mp4   # Video de prueba con detecciones
    └── log_cnn.txt                # Log de entrenamiento
```

---

## Requisitos e Instalación

1. **Requisitos del sistema:** Python 3.10 o superior.
2. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/CesarMauriciof/deteccion-pico-placa-popayan.git
   cd deteccion-pico-placa-popayan
   ```
3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
4. **(Opcional) OCR clásico de comparación:** instalar Tesseract y `pip install pytesseract`.
5. **(Opcional) Asistente generativo:** configurar la clave de API.
   ```bash
   # Windows (PowerShell)
   $env:ANTHROPIC_API_KEY="sk-ant-..."
   # Linux / macOS
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```
   Sin clave, el asistente funciona en modo de respaldo (sin internet).

El modelo `modelos/cnn_caracteres.keras` ya viene entrenado; solo es necesario reentrenar si se quieren modificar los datos o la arquitectura.

---

## Guía de Uso

| Paso | Archivo | Qué hace |
| :--- | :--- | :--- |
| 1 (opcional) | `entrenar_cnn_caracteres.py` | Genera datos sintéticos y entrena la CNN. |
| 2 | `app_tiempo_real.py` | Cámara web o video en tiempo real. |
| 3 | `asistente.py` | Chat en consola con el asistente. |
| 4 (opcional) | `evaluar_sistema.py` | Mide detección, lectura, falsos positivos y velocidad. |
| - | `crear_video_demo.py` | Crea un video de prueba si no hay cámara. |
| - | `demo_proyecto.ipynb` | Recorrido paso a paso en Jupyter. |

### Aplicación en tiempo real

```bash
python app_tiempo_real.py             # cámara web
python app_tiempo_real.py video.mp4   # archivo de video
```

Teclas: `q` salir · `a` preguntar al asistente · `s` guardar captura.

- Recuadro **verde** = puede circular · recuadro **rojo** = tiene restricción.
- Cada placa confirmada se registra en `resultados/registro_detecciones.csv`, que el asistente puede consultar.

### Clasificar una foto

```python
import cv2
from lector_placas import LectorPlacas
import pico_placa as pyp

lector = LectorPlacas()
for r in lector.leer(cv2.imread("mi_foto.jpg")):
    print(r["texto"], pyp.verificar(r["texto"])["explicacion"])
```

### Entrenar la CNN

```bash
python entrenar_cnn_caracteres.py
```

### Evaluar el sistema

```bash
python evaluar_sistema.py
```

---

## Lecciones Aprendidas y Mejoras Futuras

1. **La detección manda:** una CNN excelente (`96%`) rinde poco si la detección y la segmentación fallan; pasar de `37%` a `80.67%` vino de mejorar OpenCV, no la red.
2. **Combinar vías de detección:** color y bordes son complementarios; usar solo color fallaba con vehículos amarillos o fondos de pasto.
3. **El conocimiento de dominio es un multiplicador:** la regla `ABC123` y la votación temporal corrigen errores que la CNN no puede resolver sola.
4. **Datos sintéticos bien diseñados:** degradar las placas con la *misma* segmentación del pipeline real cierra la brecha entre entrenamiento y uso.
5. **Diseño desacoplado:** separar reglas (`config_popayan.json`), visión (`placas_core.py`) y lógica de negocio (`pico_placa.py`) permite actualizar la normativa sin tocar el código.

**Mejoras futuras:**

- Entrenar con GPU para escalar épocas y filtros.
- Añadir *label smoothing* y corregir el sesgo de confusión entre dígitos.
- Soporte para placas de taxi (blancas) y motos, ampliando la detección por color.
- Despliegue del asistente como API web y almacenamiento persistente de detecciones.
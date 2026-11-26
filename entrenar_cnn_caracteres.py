# -*- coding: utf-8 -*-
"""
entrenar_cnn_caracteres.py - CNN desde cero para leer caracteres de placas
==========================================================================
Misma lógica del taller de CIFAR-10, adaptada a 36 clases (0-9 y A-Z):
    1. Generar datos (placas sintéticas -> segmentación -> recortes 32x32)
    2. Definir la CNN (3 bloques convolucionales + capa densa)
    3. Entrenar con data augmentation, dropout y reducción del learning rate
    4. Evaluar con un conjunto de prueba generado con otra semilla
No se usa ningún modelo preentrenado.

Ejecutar por celdas en Spyder (Ctrl+Enter) o completo (F5).
Tarda unos minutos en CPU. Al terminar guarda modelos/cnn_caracteres.keras
"""

# %% 0. LIBRERÍAS
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import confusion_matrix

import generador_sintetico as gs

tf.keras.utils.set_random_seed(42)
os.makedirs("modelos", exist_ok=True)
os.makedirs("resultados", exist_ok=True)
CLASES = gs.CLASES                                   # "0123456789ABC...Z"

# %% 1. DATOS SINTÉTICOS
t0 = time.time()
print("Generando datos de ENTRENAMIENTO...")
X_tr, y_tr, f_tr = gs.generar_dataset_caracteres(n_placas=12000, semilla=1)
print("Generando datos de PRUEBA (otra semilla, placas nunca vistas)...")
X_te, y_te, f_te = gs.generar_dataset_caracteres(n_placas=2000, semilla=999)
print(f"Entrenamiento: {X_tr.shape} | Prueba: {X_te.shape} | "
      f"{(time.time() - t0) / 60:.1f} min")

# Separar validación (10 %) del entrenamiento
idx = np.random.RandomState(0).permutation(len(X_tr))
n_val = len(X_tr) // 10
X_val, y_val = X_tr[idx[:n_val]], y_tr[idx[:n_val]]
X_tr, y_tr = X_tr[idx[n_val:]], y_tr[idx[n_val:]]

conteo = np.bincount(y_tr, minlength=len(CLASES))
print("Ejemplos por clase (mín / máx):", conteo.min(), "/", conteo.max())

plt.figure(figsize=(10, 3))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(X_tr[i, :, :, 0], cmap="gray")
    plt.title(CLASES[y_tr[i]], fontsize=9)
    plt.axis("off")
plt.tight_layout()
plt.savefig("resultados/muestras_entrenamiento.png", dpi=110)
plt.show()

# %% 2. ARQUITECTURA
AUMENTO = tf.keras.Sequential([
    layers.RandomTranslation(0.08, 0.08, fill_mode="constant"),
    layers.RandomRotation(0.03, fill_mode="constant"),     # ±11°
    layers.RandomZoom(0.1, fill_mode="constant"),
], name="data_augmentation")


def bloque(x, filtros, dropout):
    for _ in range(2):
        x = layers.Conv2D(filtros, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    return layers.Dropout(dropout)(x)


def construir_cnn(n_clases=36):
    entrada = layers.Input(shape=(32, 32, 1))
    x = AUMENTO(entrada)                                   # solo actúa al entrenar
    x = bloque(x, 32, 0.15)                                # 32 -> 16
    x = bloque(x, 64, 0.20)                                # 16 -> 8
    x = bloque(x, 128, 0.25)                               # 8  -> 4
    x = layers.Flatten()(x)
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)
    salida = layers.Dense(n_clases, activation="softmax")(x)
    return models.Model(entrada, salida, name="CNN_caracteres_placas")


modelo = construir_cnn(len(CLASES))
modelo.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
               loss="sparse_categorical_crossentropy", metrics=["accuracy"])
modelo.summary()

# %% 3. ENTRENAMIENTO
cbs = [
    callbacks.ReduceLROnPlateau(monitor="val_accuracy", mode="max", factor=0.5,
                                patience=2, min_lr=1e-5, verbose=1),
    callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                            restore_best_weights=True, verbose=1),
]
t0 = time.time()
hist = modelo.fit(X_tr, y_tr, batch_size=128, epochs=20,
                  validation_data=(X_val, y_val), callbacks=cbs, verbose=2)
minutos = (time.time() - t0) / 60

# %% 4. EVALUACIÓN
perdida, precision = modelo.evaluate(X_te, y_te, verbose=0)
pred = np.argmax(modelo.predict(X_te, verbose=0), axis=1)
print(f"\n>>> Precisión por carácter en prueba: {precision:.4f}")

cm = confusion_matrix(y_te, pred, labels=range(len(CLASES)))
errores = [(CLASES[i], CLASES[j], int(cm[i, j]))
           for i in range(len(CLASES)) for j in range(len(CLASES))
           if i != j and cm[i, j] > 0]
errores.sort(key=lambda e: -e[2])
print("Confusiones más frecuentes (real -> predicho, veces):", errores[:10])

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(hist.history["accuracy"], label="Entrenamiento")
ax[0].plot(hist.history["val_accuracy"], label="Validación")
ax[0].set_title("Precisión"); ax[0].set_xlabel("Época"); ax[0].legend()
ax[1].plot(hist.history["loss"], label="Entrenamiento")
ax[1].plot(hist.history["val_loss"], label="Validación")
ax[1].set_title("Pérdida"); ax[1].set_xlabel("Época"); ax[1].legend()
plt.tight_layout()
plt.savefig("resultados/curvas_cnn_caracteres.png", dpi=110)
plt.show()

modelo.save("modelos/cnn_caracteres.keras")
with open("resultados/metricas_cnn.json", "w", encoding="utf-8") as f:
    json.dump({"precision_prueba": float(precision), "perdida_prueba": float(perdida),
               "n_entrenamiento": int(len(X_tr)), "n_validacion": int(len(X_val)),
               "n_prueba": int(len(X_te)), "epocas": len(hist.history["loss"]),
               "minutos": minutos, "parametros": int(modelo.count_params()),
               "segmentacion_fallida_entrenamiento": int(f_tr),
               "confusiones": errores[:15],
               "historial": {k: [float(v) for v in vs] for k, vs in hist.history.items()}},
              f, indent=2, ensure_ascii=False)
print("Modelo guardado en modelos/cnn_caracteres.keras")

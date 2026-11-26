# -*- coding: utf-8 -*-
"""
lector_placas.py - Pipeline completo: imagen -> texto de la placa
=================================================================
    1. detectar_placas()       (OpenCV, color amarillo + forma)
    2. rectificar()            (perspectiva -> 330 x 160)
    3. segmentar_caracteres()  (6 recortes de 32 x 32)
    4. CNN propia              (36 clases: 0-9, A-Z)
    5. Regla de formato        (carro particular: 3 letras + 3 números)

La regla de formato es conocimiento del dominio: en las 3 primeras
posiciones solo pueden ir letras y en las 3 últimas solo números. Se
aplica sobre las probabilidades de la CNN, lo que corrige confusiones
típicas como O/0, I/1, B/8, S/5 o Z/2.

Uso:
    from lector_placas import LectorPlacas
    lector = LectorPlacas()
    resultados = lector.leer(imagen_bgr)
"""

import os
from collections import Counter, deque

import cv2
import numpy as np
import tensorflow as tf

import placas_core as pc

CLASES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ES_DIGITO = np.array([c.isdigit() for c in CLASES])
RUTA_MODELO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "modelos", "cnn_caracteres.keras")


class LectorPlacas:
    def __init__(self, ruta_modelo=RUTA_MODELO, usar_formato=True):
        self.modelo = tf.keras.models.load_model(ruta_modelo)
        self.usar_formato = usar_formato

    # ------------------------------------------------------------------
    def reconocer_caracteres(self, recortes):
        """Devuelve (texto, confianza_mínima, probabilidades)."""
        X = np.stack(recortes)[..., None].astype("float32")
        probs = self.modelo(X, training=False).numpy()
        if self.usar_formato and len(recortes) == 6:
            mascara = np.zeros_like(probs)
            mascara[:3, ~ES_DIGITO] = 1          # posiciones 1-3: letras
            mascara[3:, ES_DIGITO] = 1           # posiciones 4-6: números
            probs = probs * mascara
            probs = probs / probs.sum(axis=1, keepdims=True)
        idx = probs.argmax(axis=1)
        texto = "".join(CLASES[i] for i in idx)
        conf = float(probs.max(axis=1).min())
        return texto, conf, probs

    def leer(self, frame, max_placas=2):
        """Lee las placas de una imagen BGR.
        Devuelve lista de dicts: texto, confianza, esquinas, placa (recorte)."""
        resultados = []
        for cand in pc.detectar_placas(frame, max_candidatos=max_placas + 2):
            placa = pc.rectificar(frame, cand["esquinas"])
            recortes = pc.segmentar_caracteres(placa)
            if len(recortes) != 6:            # no parece una placa legible
                continue
            texto, conf, _ = self.reconocer_caracteres(recortes)
            resultados.append({"texto": texto, "confianza": conf,
                               "esquinas": cand["esquinas"], "placa": placa})
            if len(resultados) >= max_placas:
                break
        return resultados


class VotacionTemporal:
    """En video, la misma placa aparece en muchos frames seguidos. Se
    acumulan las lecturas recientes y se reporta la más votada: así un
    frame borroso no cambia el resultado."""

    def __init__(self, ventana=12, minimo_votos=4):
        self.lecturas = deque(maxlen=ventana)
        self.minimo_votos = minimo_votos

    def agregar(self, texto):
        self.lecturas.append(texto)
        texto_top, votos = Counter(self.lecturas).most_common(1)[0]
        return texto_top if votos >= self.minimo_votos else None

    def reiniciar(self):
        self.lecturas.clear()


def dibujar_resultado(frame, esquinas, texto, color=(0, 200, 0), estado=None):
    """Dibuja el contorno de la placa y su lectura sobre el frame."""
    pts = esquinas.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], True, color, 3)
    x, y = int(esquinas[:, 0].min()), int(esquinas[:, 1].min())
    etiqueta = texto if estado is None else f"{texto}  {estado}"
    (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (x, y - th - 14), (x + tw + 10, y), color, -1)
    cv2.putText(frame, etiqueta, (x + 5, y - 7), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

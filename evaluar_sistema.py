# -*- coding: utf-8 -*-
"""
evaluar_sistema.py - Evaluación del sistema completo con respuesta conocida
==========================================================================
Genera escenas de prueba pegando placas sintéticas (texto y posición
conocidos) sobre fotos reales de vehículos y calles (carpeta fondos/),
con perspectiva, sombras, desenfoque y compresión. Mide:

  - Detección: ¿encontró la placa? (IoU >= 0,5 con la posición real)
  - Segmentación: ¿separó los 6 caracteres?
  - Lectura: caracteres y placas completas correctas
  - CNN propia vs. Tesseract OCR (motor clásico de OCR) sobre la misma placa
  - Con y sin la regla de formato (3 letras + 3 números)
  - Falsos positivos en fotos SIN placa (taxis, buses y objetos amarillos)
  - Velocidad (cuadros por segundo)
"""

# %% 0. LIBRERÍAS
import os
import glob
import json
import time
import random

import cv2
import numpy as np

import placas_core as pc
import generador_sintetico as gs
from lector_placas import LectorPlacas, dibujar_resultado

try:
    import pytesseract
    TESSERACT = True
except ImportError:
    TESSERACT = False

N_ESCENAS, N_NEGATIVAS, SEMILLA = 300, 51, 2026
rng = random.Random(SEMILLA)
np.random.seed(SEMILLA)


def cargar_fondo(ruta, ancho=960, alto=720):
    img = cv2.imread(ruta)
    h, w = img.shape[:2]
    esc = max(ancho / w, alto / h)
    img = cv2.resize(img, (int(w * esc) + 1, int(h * esc) + 1))
    y0, x0 = (img.shape[0] - alto) // 2, (img.shape[1] - ancho) // 2
    return img[y0:y0 + alto, x0:x0 + ancho]


def iou_poligonos(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return inter / union if union > 0 else 0.0


def leer_tesseract(placa):
    """OCR clásico sobre la franja de caracteres de la placa rectificada."""
    franja = cv2.cvtColor(placa[10:120, 8:-8], cv2.COLOR_BGR2GRAY)
    franja = cv2.threshold(franja, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    txt = pytesseract.image_to_string(
        franja, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    return "".join(c for c in txt.upper() if c.isalnum())


# %% 1. ESCENAS CON PLACA
fondos = sorted(glob.glob("fondos/*.JPEG") + glob.glob("fondos/*.jpg"))
lector = LectorPlacas(usar_formato=True)
lector_sin = LectorPlacas(usar_formato=False)
lector_sin.modelo = lector.modelo                         # mismo modelo, sin regla

m = dict(detectadas=0, segmentadas=0, placas_ok=0, placas_ok_sin_formato=0,
         car_ok=0, car_total=0, tess_ok=0, tess_car_ok=0, tiempos=[])
errores, ejemplos = [], []
for i in range(N_ESCENAS):
    fondo = cargar_fondo(rng.choice(fondos))
    texto = gs.placa_aleatoria(rng)
    escena, esq_real = gs.componer_escena(fondo, texto, rng)

    t0 = time.perf_counter()
    cands = pc.detectar_placas(escena, max_candidatos=4)
    lect = lector.leer(escena, max_placas=1)
    m["tiempos"].append(time.perf_counter() - t0)

    det = [c for c in cands if iou_poligonos(c["esquinas"], esq_real) >= 0.5]
    if not det:
        errores.append((texto, "no detectada"))
        continue
    m["detectadas"] += 1
    placa = pc.rectificar(escena, det[0]["esquinas"])
    recortes = pc.segmentar_caracteres(placa)
    if len(recortes) != 6:
        errores.append((texto, f"segmentación ({len(recortes)} caracteres)"))
        continue
    m["segmentadas"] += 1
    leido, _, _ = lector.reconocer_caracteres(recortes)
    leido_sin, _, _ = lector_sin.reconocer_caracteres(recortes)
    m["placas_ok"] += leido == texto
    m["placas_ok_sin_formato"] += leido_sin == texto
    m["car_ok"] += sum(a == b for a, b in zip(leido, texto))
    m["car_total"] += 6
    if leido != texto:
        errores.append((texto, f"lectura: {leido}"))
    if TESSERACT:
        t = leer_tesseract(placa)
        m["tess_ok"] += t == texto
        m["tess_car_ok"] += sum(a == b for a, b in zip(t.ljust(6)[:6], texto))
    if len(ejemplos) < 8 and lect:
        vis = escena.copy()
        dibujar_resultado(vis, lect[0]["esquinas"], lect[0]["texto"],
                          (0, 190, 0) if lect[0]["texto"] == texto else (0, 0, 230))
        ejemplos.append(cv2.resize(vis, (480, 360)))

# %% 2. ESCENAS SIN PLACA (falsos positivos)
falsos = 0
for ruta in fondos[:N_NEGATIVAS]:
    if lector.leer(cargar_fondo(ruta), max_placas=1):
        falsos += 1

# %% 3. RESUMEN
N = N_ESCENAS
res = {
    "escenas": N,
    "deteccion": m["detectadas"] / N,
    "segmentacion_sobre_detectadas": m["segmentadas"] / max(1, m["detectadas"]),
    "precision_caracter_cnn": m["car_ok"] / max(1, m["car_total"]),
    "placas_correctas_sobre_segmentadas": m["placas_ok"] / max(1, m["segmentadas"]),
    "placas_correctas_sin_formato_sobre_segmentadas":
        m["placas_ok_sin_formato"] / max(1, m["segmentadas"]),
    "tesseract_placas_correctas_sobre_segmentadas":
        m["tess_ok"] / max(1, m["segmentadas"]) if TESSERACT else None,
    "tesseract_precision_caracter":
        m["tess_car_ok"] / max(1, m["car_total"]) if TESSERACT else None,
    "sistema_completo_placas_correctas": m["placas_ok"] / N,
    "imagenes_sin_placa": N_NEGATIVAS,
    "falsos_positivos": falsos,
    "ms_por_imagen": 1000 * float(np.mean(m["tiempos"])),
    "fps": 1 / float(np.mean(m["tiempos"])),
    "tipos_de_error": {},
}
for _, e in errores:
    k = e.split(":")[0].split(" (")[0]
    res["tipos_de_error"][k] = res["tipos_de_error"].get(k, 0) + 1
res["ejemplos_error"] = errores[:15]

for k, v in res.items():
    print(f"{k:<50} {v:.3f}" if isinstance(v, float) else f"{k:<50} {v}")
os.makedirs("resultados", exist_ok=True)
with open("resultados/metricas_sistema.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
if ejemplos:
    filas = [np.hstack(ejemplos[j:j + 4]) for j in range(0, len(ejemplos) - 3, 4)]
    cv2.imwrite("resultados/ejemplos_deteccion.jpg", np.vstack(filas))

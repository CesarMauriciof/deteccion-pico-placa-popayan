# -*- coding: utf-8 -*-
"""
crear_video_demo.py - Video de prueba para la aplicación en tiempo real
=======================================================================
Si no hay cámara a mano, este script arma un video corto en el que dos
"vehículos" se acercan a la cámara uno después del otro: el primero con
una placa SIN restricción hoy y el segundo CON restricción (según la
regla del día en que se ejecute). Luego se puede probar:

    python app_tiempo_real.py resultados/video_demo.mp4
"""

import random
from datetime import date

import cv2
import numpy as np

import generador_sintetico as gs
import pico_placa as pyp


def placa_para(restringida, rng):
    """Crea una placa que tenga (o no) pico y placa hoy."""
    digitos_hoy = pyp.digitos_del_dia(date.today())
    while True:
        p = gs.placa_aleatoria(rng)
        if (int(p[-1]) in digitos_hoy) == restringida:
            return p


def crear_video(ruta="resultados/video_demo.mp4", fondo_ruta="fondos/n03770679_minivan.JPEG",
                segundos_por_placa=4, fps=20, semilla=5):
    rng = random.Random(semilla)
    fondo = cv2.resize(cv2.imread(fondo_ruta), (960, 720))
    placas = [placa_para(False, rng), placa_para(True, rng)]
    escritor = cv2.VideoWriter(ruta, cv2.VideoWriter_fourcc(*"mp4v"), fps, (960, 720))
    for texto in placas:
        imagen_placa = gs.dibujar_placa(texto, ciudad="POPAYAN", rng=rng)
        n = segundos_por_placa * fps
        for i in range(n):
            t = i / (n - 1)
            ancho = int(110 + 170 * t)                 # el vehículo se acerca
            alto = int(ancho * 160 / 330)
            x = int(420 - ancho / 2 + 60 * np.sin(t * 3))
            y = int(430 + 80 * t)
            frame = fondo.copy()
            peq = cv2.resize(imagen_placa, (ancho, alto), interpolation=cv2.INTER_AREA)
            frame[y:y + alto, x:x + ancho] = peq
            frame = gs.degradar(frame, rng, fuerte=0.4)
            escritor.write(frame)
        for _ in range(fps // 2):                     # medio segundo sin placa
            escritor.write(fondo)
    escritor.release()
    print("Video creado:", ruta, "| placas:", placas)
    return placas


if __name__ == "__main__":
    crear_video()

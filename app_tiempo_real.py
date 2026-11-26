# -*- coding: utf-8 -*-
"""
app_tiempo_real.py - Detección de placas y pico y placa en tiempo real
======================================================================
Abre la cámara web (o un video), detecta y lee las placas, verifica el
pico y placa de Popayán y lo muestra sobre la imagen:
    recuadro VERDE = puede circular | recuadro ROJO = tiene restricción

Cada placa confirmada (leída igual en varios frames seguidos) se guarda
en resultados/registro_detecciones.csv, que el asistente puede consultar.

Teclas (con la ventana de video activa):
    q  salir
    a  hacer una pregunta al asistente (se escribe en la consola)
    s  guardar una captura en resultados/

Ejecutar:  python app_tiempo_real.py            (cámara 0)
           python app_tiempo_real.py video.mp4  (archivo de video)
"""

import os
import sys
import time
from datetime import datetime

import cv2

from lector_placas import LectorPlacas, VotacionTemporal, dibujar_resultado
import pico_placa as pyp
from asistente import Asistente, registrar_deteccion

FUENTE = sys.argv[1] if len(sys.argv) > 1 else 0     # 0 = cámara web
ANCHO_PROCESO = 960                                  # se reduce el frame para ir rápido
SEGUNDOS_ENTRE_REGISTROS = 30                        # no repetir la misma placa seguido
GUARDAR_VIDEO = os.environ.get("GUARDAR_VIDEO")      # ruta .mp4 opcional
MOSTRAR = os.environ.get("SIN_VENTANA") is None


def main(fuente=FUENTE, max_frames=None):
    lector = LectorPlacas()
    asistente = Asistente()
    votacion = VotacionTemporal(ventana=12, minimo_votos=4)
    ultimo_registro = {}
    cap = cv2.VideoCapture(int(fuente) if str(fuente).isdigit() else fuente)
    if not cap.isOpened():
        print("No se pudo abrir la fuente de video:", fuente)
        return
    print(f"Asistente en modo {asistente.modo}. Teclas: q salir | a preguntar | s captura")

    escritor, n, t0, fps = None, 0, time.time(), 0.0
    confirmada, estado_txt, color = None, "", (0, 200, 0)
    frames_sin_placa = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        esc = ANCHO_PROCESO / frame.shape[1]
        if esc < 1:
            frame = cv2.resize(frame, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)

        lecturas = lector.leer(frame, max_placas=1)
        if lecturas:
            frames_sin_placa = 0
            r = lecturas[0]
            votada = votacion.agregar(r["texto"])
            if votada:
                if votada != confirmada:
                    confirmada = votada
                    v = pyp.verificar(votada)
                    color = (0, 0, 230) if v["restringido"] else (0, 190, 0)
                    estado_txt = "PICO Y PLACA" if v["restringido"] else "PUEDE CIRCULAR"
                    print(f"[{datetime.now():%H:%M:%S}] {v['explicacion']}")
                    if time.time() - ultimo_registro.get(votada, 0) > SEGUNDOS_ENTRE_REGISTROS:
                        registrar_deteccion(votada, v["restringido"], r["confianza"])
                        ultimo_registro[votada] = time.time()
                dibujar_resultado(frame, r["esquinas"], confirmada, color, estado_txt)
            else:
                dibujar_resultado(frame, r["esquinas"], "leyendo...", (0, 180, 255))
        else:
            frames_sin_placa += 1
            if frames_sin_placa > 15:            # la placa salió de la escena
                votacion.reiniciar()
                confirmada = None

        if n % 10 == 0:
            fps = 10 / (time.time() - t0)
            t0 = time.time()
        cv2.putText(frame, f"{fps:4.1f} FPS | Pico y placa Popayan | {datetime.now():%a %H:%M}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        if GUARDAR_VIDEO:
            if escritor is None:
                h, w = frame.shape[:2]
                escritor = cv2.VideoWriter(GUARDAR_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"),
                                           20, (w, h))
            escritor.write(frame)
        if MOSTRAR:
            cv2.imshow("Placas - Pico y placa", frame)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            if tecla == ord("s"):
                ruta = f"resultados/captura_{datetime.now():%H%M%S}.png"
                cv2.imwrite(ruta, frame)
                print("Captura guardada:", ruta)
            if tecla == ord("a"):
                pregunta = input("Pregunta para el asistente: ")
                print("Asistente:", asistente.preguntar(pregunta))
        if max_frames and n >= max_frames:
            break

    cap.release()
    if escritor:
        escritor.release()
    if MOSTRAR:
        cv2.destroyAllWindows()
    return n


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
placas_core.py - Visión computacional clásica (OpenCV) para placas colombianas
==============================================================================
Funciones de bajo nivel que usan los demás módulos:

    detectar_placas(frame)      -> lista de candidatos (4 esquinas + puntaje)
    rectificar(frame, esquinas) -> placa "de frente" de 330 x 160 píxeles
    segmentar_caracteres(placa) -> recortes de los 6 caracteres, de izquierda
                                   a derecha, listos para la CNN (32 x 32)

Idea principal: las placas de vehículos PARTICULARES en Colombia son
AMARILLAS con caracteres negros. Por eso la detección empieza buscando
regiones amarillas en el espacio de color HSV, y luego filtra por forma
(proporción 33 x 16 cm, es decir, ancho / alto ~ 2,06).
"""

import cv2
import numpy as np

# Tamaño estándar al que se "endereza" cada placa (proporción real 33 x 16 cm)
ANCHO_PLACA, ALTO_PLACA = 330, 160
TAM_CARACTER = 32                      # entrada de la CNN: 32 x 32 en gris

# Rango de amarillo en HSV (OpenCV usa H de 0 a 179)
AMARILLO_BAJO = np.array([15, 100, 60])
AMARILLO_ALTO = np.array([36, 255, 255])


# ----------------------------------------------------------------------
# 1. DETECCIÓN
# ----------------------------------------------------------------------
def _ordenar_esquinas(pts):
    """Ordena 4 puntos como: sup-izq, sup-der, inf-der, inf-izq."""
    pts = np.asarray(pts, dtype="float32").reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")


def mascara_amarilla(frame):
    """Devuelve una máscara binaria con las zonas amarillas de la imagen."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mascara = cv2.inRange(hsv, AMARILLO_BAJO, AMARILLO_ALTO)
    # Cierre morfológico: une el fondo amarillo que los caracteres negros
    # "parten" en pedazos, para que la placa sea una sola región.
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, k, iterations=1)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    return mascara


def _verificar_color(frame, esquinas):
    """Endereza el candidato a 110 x 54 y mide qué fracción es amarilla y
    qué fracción es oscura (caracteres). Una placa real tiene mucho
    amarillo y entre 8 % y 50 % de píxeles oscuros."""
    destino = np.float32([[0, 0], [109, 0], [109, 53], [0, 53]])
    M = cv2.getPerspectiveTransform(_ordenar_esquinas(esquinas), destino)
    peq = cv2.warpPerspective(frame, M, (110, 54))
    hsv = cv2.cvtColor(peq, cv2.COLOR_BGR2HSV)
    interior = hsv[4:-4, 4:-4]
    h, s, v = interior[..., 0], interior[..., 1], interior[..., 2]
    oscuro = v < np.percentile(v, 90) * 0.55
    amarillo = (h >= 12) & (h <= 38) & (s >= 60) & ~oscuro
    claro = ~oscuro
    frac_amarillo = amarillo.sum() / max(1, claro.sum())   # entre lo no oscuro
    frac_oscuro = oscuro.mean()
    return frac_amarillo, frac_oscuro


def _forma(pts):
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts.astype(np.float32))
    if w < h:
        w, h = h, w
    return (w / h if h > 0 else 0), w * h


def _iou(a, b):
    inter, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    u = cv2.contourArea(a.astype(np.float32)) + cv2.contourArea(b.astype(np.float32)) - inter
    return inter / u if u > 0 else 0.0


def _cuadrilateros_por_bordes(frame, area_min):
    """Vía 2: cuadriláteros cerrados en el mapa de bordes. El marco negro
    de la placa produce un contorno de 4 lados aunque el vehículo también
    sea amarillo o la placa esté rodeada de pasto."""
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)
    med = np.median(gris)
    bordes = cv2.Canny(gris, int(max(0, 0.5 * med)), int(min(255, 1.3 * med)))
    bordes = cv2.dilate(bordes, np.ones((2, 2), np.uint8))
    contornos, _ = cv2.findContours(bordes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    salida = []
    for c in contornos:
        if cv2.contourArea(c) < area_min:
            continue
        peri = cv2.arcLength(c, True)
        for eps in (0.02, 0.04):
            aprox = cv2.approxPolyDP(c, eps * peri, True)
            if len(aprox) == 4 and cv2.isContourConvex(aprox):
                salida.append(_ordenar_esquinas(aprox))
                break
    return salida


def _cuadrilateros_por_color(frame, area_min):
    """Vía 1: regiones amarillas (máscara HSV) con forma de rectángulo."""
    mascara = mascara_amarilla(frame)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    salida = []
    for c in contornos:
        area = cv2.contourArea(c)
        if area < area_min:
            continue
        rect = cv2.minAreaRect(c)
        w, h = rect[1]
        if w * h == 0 or area / (w * h) < 0.55:     # qué tan "rectangular" es
            continue
        peri = cv2.arcLength(c, True)
        aprox = cv2.approxPolyDP(cv2.convexHull(c), 0.04 * peri, True)
        salida.append(_ordenar_esquinas(aprox) if len(aprox) == 4
                      else _ordenar_esquinas(cv2.boxPoints(rect)))
    return salida


def detectar_placas(frame, area_min_rel=0.0012, max_candidatos=3):
    """Busca placas amarillas en un frame BGR combinando dos vías
    (color y bordes) y verificando cada candidato por su contenido.

    Devuelve una lista de diccionarios ordenada por puntaje:
        {"esquinas": array 4x2, "rect": (x, y, w, h), "puntaje": float}
    """
    alto, ancho = frame.shape[:2]
    area_min = area_min_rel * alto * ancho
    candidatos = []
    for esquinas in (_cuadrilateros_por_color(frame, area_min) +
                     _cuadrilateros_por_bordes(frame, area_min)):
        proporcion, area = _forma(esquinas)
        if not (1.4 <= proporcion <= 3.6) or area > 0.5 * alto * ancho:
            continue
        frac_amarillo, frac_oscuro = _verificar_color(frame, esquinas)
        if frac_amarillo < 0.55 or not (0.06 <= frac_oscuro <= 0.55):
            continue
        # Puntaje: color + cercanía a la proporción real (2,06)
        puntaje = frac_amarillo * np.exp(-abs(proporcion - 2.06))
        candidatos.append({"esquinas": esquinas,
                           "rect": cv2.boundingRect(esquinas.astype(np.int32)),
                           "puntaje": float(puntaje)})
    # Supresión de no máximos: si dos candidatos se superponen, queda el mejor
    candidatos.sort(key=lambda d: d["puntaje"], reverse=True)
    finales = []
    for c in candidatos:
        if all(_iou(c["esquinas"], f["esquinas"]) < 0.3 for f in finales):
            finales.append(c)
        if len(finales) >= max_candidatos:
            break
    return finales


def expandir(esquinas, factor=0.04):
    """Agranda el cuadrilátero un poco hacia afuera desde su centro, para
    no cortar los caracteres si el detector encontró el borde interior."""
    c = esquinas.mean(axis=0)
    return (c + (esquinas - c) * (1 + factor)).astype("float32")


def rectificar(frame, esquinas, margen=0.04):
    """Transformación de perspectiva: deja la placa de frente (330 x 160)."""
    esquinas = expandir(_ordenar_esquinas(esquinas), margen)
    destino = np.array([[0, 0], [ANCHO_PLACA - 1, 0],
                        [ANCHO_PLACA - 1, ALTO_PLACA - 1], [0, ALTO_PLACA - 1]],
                       dtype="float32")
    M = cv2.getPerspectiveTransform(_ordenar_esquinas(esquinas), destino)
    return cv2.warpPerspective(frame, M, (ANCHO_PLACA, ALTO_PLACA))


# ----------------------------------------------------------------------
# 2. SEGMENTACIÓN DE CARACTERES
# ----------------------------------------------------------------------
def binarizar_placa(placa):
    """Caracteres en blanco sobre fondo negro (lo que necesita la CNN)."""
    gris = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (3, 3), 0)
    # Umbral adaptativo: tolera sombras y reflejos sobre la placa
    binaria = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 10)
    return binaria


def quitar_marco(binaria):
    """Borra las líneas largas del marco de la placa (horizontales y
    verticales) con aperturas morfológicas, para que el marco no se pegue
    a los caracteres."""
    h, w = binaria.shape
    hor = cv2.morphologyEx(binaria, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.3), 1)))
    ver = cv2.morphologyEx(binaria, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h * 0.85))))
    lineas = cv2.dilate(hor | ver, np.ones((3, 3), np.uint8))
    return cv2.bitwise_and(binaria, cv2.bitwise_not(lineas))


def segmentar_caracteres(placa, n_esperado=6, devolver_cajas=False):
    """Encuentra los caracteres de la franja superior de la placa.

    Usa componentes conectados y filtra por tamaño y posición: un carácter
    ocupa entre el 22 % y el 85 % de la altura de la placa y su centro está
    en la parte superior. El nombre de la ciudad (letras pequeñas abajo),
    el separador y el marco quedan descartados.
    Devuelve una lista de arreglos 32 x 32 (float, 0-1).
    """
    binaria = quitar_marco(binarizar_placa(placa))
    m = 3
    binaria[:m, :] = 0; binaria[-m:, :] = 0
    binaria[:, :m] = 0; binaria[:, -m:] = 0

    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(binaria, 8)
    cajas = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (0.22 * ALTO_PLACA <= h <= 0.85 * ALTO_PLACA):
            continue
        if not (0.012 * ANCHO_PLACA <= w <= 0.20 * ANCHO_PLACA):
            continue                                 # la "I" y el "1" son angostos
        if area < 0.12 * w * h or (y + h / 2) > 0.72 * ALTO_PLACA:
            continue
        if x < 0.035 * ANCHO_PLACA or x + w > 0.965 * ANCHO_PLACA:
            continue                                 # restos del marco lateral
        cajas.append((x, y, w, h))

    # Si hay más de 6, se quedan los 6 de altura más parecida a la de los
    # caracteres más altos (los de la placa son todos del mismo tamaño)
    cajas.sort(key=lambda c: c[0])                   # de izquierda a derecha
    if len(cajas) > n_esperado:
        # Se elige la ventana de 6 cajas consecutivas más pareja en altura
        # y en posición vertical (los 6 caracteres están alineados)
        def costo(grupo):
            hs = np.array([c[3] for c in grupo], float)
            ys = np.array([c[1] for c in grupo], float)
            return hs.std() + ys.std()
        inicio = min(range(len(cajas) - n_esperado + 1),
                     key=lambda i: costo(cajas[i:i + n_esperado]))
        cajas = cajas[inicio:inicio + n_esperado]

    recortes = [normalizar_caracter(binaria[y:y + h, x:x + w])
                for x, y, w, h in cajas]
    if devolver_cajas:
        return recortes, cajas, binaria
    return recortes


def normalizar_caracter(recorte):
    """Centra el carácter en un cuadrado con margen y lo lleva a 32 x 32,
    conservando su proporción (una "I" no se estira hasta ser un bloque)."""
    h, w = recorte.shape[:2]
    lado = int(max(h, w) * 1.2) + 2
    lienzo = np.zeros((lado, lado), dtype=np.uint8)
    y0, x0 = (lado - h) // 2, (lado - w) // 2
    lienzo[y0:y0 + h, x0:x0 + w] = recorte
    peq = cv2.resize(lienzo, (TAM_CARACTER, TAM_CARACTER),
                     interpolation=cv2.INTER_AREA)
    return peq.astype("float32") / 255.0

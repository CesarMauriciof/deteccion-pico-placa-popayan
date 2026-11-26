# -*- coding: utf-8 -*-
"""
generador_sintetico.py - Placas colombianas sintéticas
======================================================
1. dibujar_placa(texto): placa amarilla con borde negro, 6 caracteres y
   el nombre de la ciudad, como las de vehículos particulares.
2. generar_dataset_caracteres(): miles de caracteres para entrenar la CNN.
   Cada placa se deforma (perspectiva, desenfoque, ruido, iluminación) y
   luego pasa por la MISMA segmentación que se usa en tiempo real, así los
   recortes de entrenamiento se parecen a los que verá la red en uso.
3. componer_escena(): pega una placa sobre una foto (para pruebas con
   respuesta conocida: sabemos dónde está la placa y qué dice).

No se usa ningún dataset ni modelo externo: los datos se generan aquí.
"""

import os
import glob
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import placas_core as pc

LETRAS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITOS = "0123456789"
CLASES = DIGITOS + LETRAS                       # 36 clases para la CNN
CIUDADES = ["POPAYAN", "POPAYAN", "POPAYAN", "CALI", "BOGOTA D.C.",
            "PASTO", "MEDELLIN", "TIMBIO", "PALMIRA", "JAMUNDI"]

CARPETA = os.path.dirname(os.path.abspath(__file__))
FUENTES = sorted(glob.glob(os.path.join(CARPETA, "fuentes", "*.ttf")))


def placa_aleatoria(rng=random):
    return ("".join(rng.choice(LETRAS) for _ in range(3)) +
            "".join(rng.choice(DIGITOS) for _ in range(3)))


def _texto_condensado(texto, fuente, alto_px, ancho_max, color=(15, 15, 15)):
    """Dibuja el texto y lo comprime horizontalmente para que quepa,
    imitando la tipografía angosta de las placas."""
    f = ImageFont.truetype(fuente, alto_px)
    caja = f.getbbox(texto)
    w, h = caja[2] - caja[0], caja[3] - caja[1]
    img = Image.new("RGBA", (w + 4, h + 4), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((2 - caja[0], 2 - caja[1]), texto, font=f, fill=color)
    if img.width > ancho_max:
        img = img.resize((int(ancho_max), img.height), Image.LANCZOS)
    return img


def dibujar_placa(texto, ciudad=None, fuente=None, rng=random):
    """Devuelve una placa BGR de 330 x 160 (dibujada al doble y reducida)."""
    E = 2
    W, H = pc.ANCHO_PLACA * E, pc.ALTO_PLACA * E
    amarillo = (rng.randint(225, 255), rng.randint(178, 215), rng.randint(0, 45))
    img = Image.new("RGB", (W, H), amarillo)
    d = ImageDraw.Draw(img)
    grosor = rng.randint(5, 8) * E
    m = rng.randint(3, 6) * E
    d.rounded_rectangle([m, m, W - m, H - m], radius=12 * E,
                        outline=(20, 20, 20), width=grosor)
    fuente = fuente or rng.choice(FUENTES)
    letras, numeros = texto[:3], texto[3:]
    alto_txt = int(H * rng.uniform(0.50, 0.58))
    sep = rng.choice(["  ", "  ", " - ", " · "])
    bloque = _texto_condensado(letras + sep + numeros, fuente, alto_txt,
                               ancho_max=W * rng.uniform(0.80, 0.88))
    x = (W - bloque.width) // 2 + rng.randint(-6, 6) * E
    y = int(H * rng.uniform(0.10, 0.15))
    img.paste(bloque, (x, y), bloque)
    ciudad = ciudad or rng.choice(CIUDADES)
    pie = _texto_condensado(ciudad, fuente, int(H * 0.13), W * 0.5)
    img.paste(pie, ((W - pie.width) // 2, int(H * 0.76)), pie)
    if rng.random() < 0.5:                     # tornillos
        for cx in (int(W * 0.16), int(W * 0.84)):
            d.ellipse([cx - 7 * E, int(H * 0.07) - 3 * E + m, cx + 7 * E,
                       int(H * 0.07) + 7 * E + m], fill=(90, 90, 90))
    img = img.resize((pc.ANCHO_PLACA, pc.ALTO_PLACA), Image.LANCZOS)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ----------------------------------------------------------------------
# Degradaciones realistas
# ----------------------------------------------------------------------
def degradar(img, rng=random, fuerte=1.0):
    """Iluminación, sombra, desenfoque, ruido y compresión JPEG."""
    img = img.astype(np.float32)
    img = img * rng.uniform(0.55, 1.25) + rng.uniform(-35, 25) * fuerte
    if rng.random() < 0.4 * fuerte:            # sombra en degradé
        h, w = img.shape[:2]
        g = np.linspace(rng.uniform(0.45, 1.0), 1.0, w)
        if rng.random() < 0.5:
            g = g[::-1]
        img *= g[None, :, None]
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.6 * fuerte:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.25 * fuerte:           # movimiento
        k = rng.randint(3, 7)
        ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1.0 / k
        img = cv2.filter2D(img, -1, ker)
    if rng.random() < 0.6:
        ruido = np.random.normal(0, rng.uniform(2, 9) * fuerte, img.shape)
        img = np.clip(img + ruido, 0, 255).astype(np.uint8)
    if rng.random() < 0.7:
        q = rng.randint(35, 90)
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img


def _deformar_placa(placa, rng=random, max_jit=0.10, escala=1.0):
    """Coloca la placa con perspectiva aleatoria en un lienzo y devuelve
    el lienzo y las 4 esquinas reales."""
    h, w = placa.shape[:2]
    pad = int(0.35 * w)
    W, H = w + 2 * pad, h + 2 * pad
    orig = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jit = lambda: rng.uniform(-max_jit, max_jit)
    dst = np.float32([[pad + jit() * w, pad + jit() * h],
                      [pad + w + jit() * w, pad + jit() * h],
                      [pad + w + jit() * w, pad + h + jit() * h],
                      [pad + jit() * w, pad + h + jit() * h]])
    ang = np.deg2rad(rng.uniform(-8, 8))
    c = dst.mean(axis=0)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    dst = ((dst - c) @ R.T + c).astype(np.float32)
    M = cv2.getPerspectiveTransform(orig, dst)
    fondo = np.full((H, W, 3), rng.randint(20, 200), np.uint8)
    lienzo = cv2.warpPerspective(placa, M, (W, H), dst=fondo,
                                 borderMode=cv2.BORDER_TRANSPARENT)
    if escala != 1.0:
        lienzo = cv2.resize(lienzo, None, fx=escala, fy=escala,
                            interpolation=cv2.INTER_AREA)
        dst = dst * escala
    return lienzo, dst


# ----------------------------------------------------------------------
# Dataset de caracteres para la CNN
# ----------------------------------------------------------------------
def generar_dataset_caracteres(n_placas=12000, semilla=0, verbose=True):
    """Devuelve X (N, 32, 32, 1) y y (N,) con índices de CLASES."""
    rng = random.Random(semilla)
    np.random.seed(semilla)
    X, y, fallidas = [], [], 0
    for i in range(n_placas):
        texto = placa_aleatoria(rng)
        placa = dibujar_placa(texto, rng=rng)
        escala = rng.uniform(0.35, 1.1)                 # placas lejanas/cercanas
        lienzo, esquinas = _deformar_placa(placa, rng, escala=escala)
        lienzo = degradar(lienzo, rng)
        # Esquinas con error, como las que estima el detector
        ruido = np.random.uniform(-0.025, 0.025, (4, 2)) * [pc.ANCHO_PLACA * escala,
                                                           pc.ALTO_PLACA * escala]
        recta = pc.rectificar(lienzo, esquinas + ruido.astype(np.float32))
        recortes = pc.segmentar_caracteres(recta)
        if len(recortes) != 6:
            fallidas += 1
            continue
        for r, ch in zip(recortes, texto):
            X.append(r); y.append(CLASES.index(ch))
        if verbose and (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{n_placas} placas | caracteres: {len(X)} "
                  f"| segmentación fallida: {fallidas}")
    X = np.array(X, dtype=np.float32)[..., None]
    return X, np.array(y, dtype=np.int64), fallidas


# ----------------------------------------------------------------------
# Escenas de prueba: placa pegada sobre una foto
# ----------------------------------------------------------------------
def componer_escena(fondo, texto, rng=random, ancho_placa=None):
    """Pega una placa deformada en la mitad inferior de la foto.
    Devuelve (imagen, esquinas_reales)."""
    H, W = fondo.shape[:2]
    placa = dibujar_placa(texto, rng=rng)
    ancho_placa = ancho_placa or rng.randint(int(W * 0.10), int(W * 0.28))
    escala = ancho_placa / pc.ANCHO_PLACA
    lienzo, esq = _deformar_placa(placa, rng, max_jit=0.08, escala=escala)
    # máscara de la placa dentro del lienzo
    masc = np.zeros(lienzo.shape[:2], np.uint8)
    cv2.fillConvexPoly(masc, esq.astype(np.int32), 255)
    h, w = lienzo.shape[:2]
    x0 = rng.randint(0, max(1, W - w)); y0 = rng.randint(int(H * 0.35), max(int(H * 0.35) + 1, H - h))
    roi = fondo[y0:y0 + h, x0:x0 + w]
    h2, w2 = roi.shape[:2]
    m = masc[:h2, :w2, None] / 255.0
    fondo = fondo.copy()
    fondo[y0:y0 + h2, x0:x0 + w2] = (lienzo[:h2, :w2] * m + roi * (1 - m)).astype(np.uint8)
    esq = esq + np.float32([x0, y0])
    return degradar(fondo, rng, fuerte=0.8), esq

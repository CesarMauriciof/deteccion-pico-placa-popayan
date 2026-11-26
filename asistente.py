# -*- coding: utf-8 -*-
"""
asistente.py - Asistente conversacional de pico y placa
=======================================================
Un agente al que se le pregunta en lenguaje natural, por ejemplo:
    "¿La placa JMK125 puede salir hoy?"
    "¿Qué placas tienen pico y placa el viernes?"
    "¿Qué placas detectó la cámara?"
    "¿Cuánto es la multa?"

Dos modos:
1. GENERATIVO (Claude API): un modelo de lenguaje grande (LLM, basado en
   la arquitectura Transformer) entiende la pregunta y decide qué
   HERRAMIENTA usar. Las herramientas son funciones de Python de este
   proyecto (verificar una placa, ver el historial de la cámara, etc.).
   El modelo nunca inventa la regla: la consulta en pico_placa.py.
   Requiere la variable de entorno ANTHROPIC_API_KEY.
2. RESPALDO (sin internet): reglas con expresiones regulares. Responde
   las mismas preguntas básicas si no hay clave o conexión.

Uso en Spyder:
    from asistente import Asistente
    a = Asistente()
    print(a.preguntar("¿Puede circular la placa ABC123 mañana a las 9?"))
"""

import os
import re
import csv
import json
from datetime import datetime, date, timedelta

import pico_placa as pyp

RUTA_REGISTRO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "resultados", "registro_detecciones.csv")
MODELO_CLAUDE = os.environ.get("MODELO_CLAUDE", "claude-haiku-5-5")


# ----------------------------------------------------------------------
# Registro de detecciones (lo escribe la aplicación de tiempo real)
# ----------------------------------------------------------------------
def registrar_deteccion(placa, restringido, confianza, ruta=RUTA_REGISTRO):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    nuevo = not os.path.exists(ruta)
    with open(ruta, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["fecha_hora", "placa", "restringido", "confianza"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), placa,
                    "si" if restringido else "no", f"{confianza:.2f}"])


def leer_historial(limite=10, ruta=RUTA_REGISTRO):
    if not os.path.exists(ruta):
        return []
    with open(ruta, encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    return filas[-limite:]


# ----------------------------------------------------------------------
# Herramientas que puede usar el agente
# ----------------------------------------------------------------------
def _fecha_hora(fecha=None, hora=None):
    ahora = datetime.now()
    d = date.fromisoformat(fecha) if fecha else ahora.date()
    if hora:
        h, m = map(int, hora.split(":"))
    else:
        h, m = (ahora.hour, ahora.minute) if d == ahora.date() else (9, 0)
    return datetime(d.year, d.month, d.day, h, m)


def herramienta_verificar_placa(placa, fecha=None, hora=None):
    r = pyp.verificar(placa, _fecha_hora(fecha, hora))
    if r.get("valida"):
        r["proximos_dias_con_restriccion"] = [
            pyp.fecha_texto(d) for d in pyp.proximos_dias(placa, n=3)]
    return r


def herramienta_digitos_del_dia(fecha=None):
    d = date.fromisoformat(fecha) if fecha else date.today()
    return {"fecha": pyp.fecha_texto(d), "digitos": pyp.digitos_del_dia(d),
            "motivo_sin_medida": pyp.motivo_sin_medida(d),
            "horario": f"{pyp.CONFIG['hora_inicio']} a {pyp.CONFIG['hora_fin']}"}


def herramienta_historial(limite=10):
    return {"detecciones": leer_historial(limite)}


def herramienta_reglas():
    cfg = {k: v for k, v in pyp.CONFIG.items() if not k.startswith("_")}
    return cfg


HERRAMIENTAS = {
    "verificar_placa": herramienta_verificar_placa,
    "digitos_del_dia": herramienta_digitos_del_dia,
    "historial_detecciones": herramienta_historial,
    "reglas_pico_y_placa": herramienta_reglas,
}

ESQUEMAS = [
    {"name": "verificar_placa",
     "description": "Indica si una placa tiene pico y placa en una fecha y hora en Popayán, "
                    "y sus próximos días con restricción.",
     "input_schema": {"type": "object", "properties": {
         "placa": {"type": "string", "description": "Placa, p. ej. ABC123 o ABC12D"},
         "fecha": {"type": "string", "description": "AAAA-MM-DD; si se omite, hoy"},
         "hora": {"type": "string", "description": "HH:MM en 24 horas; opcional"}},
         "required": ["placa"]}},
    {"name": "digitos_del_dia",
     "description": "Dígitos finales de placa con restricción en una fecha.",
     "input_schema": {"type": "object", "properties": {
         "fecha": {"type": "string", "description": "AAAA-MM-DD; si se omite, hoy"}}}},
    {"name": "historial_detecciones",
     "description": "Últimas placas detectadas por la cámara del sistema, con fecha, "
                    "hora y si estaban restringidas.",
     "input_schema": {"type": "object", "properties": {
         "limite": {"type": "integer", "description": "Cuántas devolver (por defecto 10)"}}}},
    {"name": "reglas_pico_y_placa",
     "description": "Reglas vigentes: rotación, horario, vehículos, sanción, excepciones y fuente.",
     "input_schema": {"type": "object", "properties": {}}},
]


def _instrucciones():
    ahora = datetime.now()
    return (
        "Eres el asistente de un sistema de reconocimiento de placas que verifica el "
        f"pico y placa en Popayán, Colombia. Hoy es {pyp.fecha_texto(ahora.date())} "
        f"y son las {ahora:%H:%M}. Responde en español, en 1 a 4 frases, de forma clara "
        "y amable. Para cualquier dato sobre restricciones, placas detectadas o reglas, "
        "usa SIEMPRE las herramientas; no inventes reglas ni fechas. Convierte expresiones "
        "como 'mañana' o 'el viernes' a una fecha AAAA-MM-DD antes de llamar a una "
        "herramienta. Si la pregunta no trata de placas, tránsito o pico y placa, dilo "
        "con amabilidad. Cuando hables de horarios o excepciones, recuerda que conviene "
        "confirmarlos con la Secretaría de Tránsito de Popayán.")


# ----------------------------------------------------------------------
# Asistente
# ----------------------------------------------------------------------
class Asistente:
    def __init__(self, usar_llm=None):
        self.cliente = None
        if usar_llm is not False and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                self.cliente = anthropic.Anthropic()
            except Exception as e:                       # sin librería o sin red
                print("No se pudo iniciar Claude, se usa el modo de respaldo:", e)
        self.historial_chat = []
        self.modo = "generativo (Claude)" if self.cliente else "respaldo sin internet"

    def preguntar(self, pregunta):
        if self.cliente:
            try:
                return self._preguntar_llm(pregunta)
            except Exception as e:
                print(f"[Claude no respondió ({e}); se usa el modo de respaldo]")
        return responder_sin_llm(pregunta)

    def _preguntar_llm(self, pregunta):
        self.historial_chat.append({"role": "user", "content": pregunta})
        mensajes = list(self.historial_chat)
        for _ in range(5):                              # máximo 5 llamadas a herramientas
            resp = self.cliente.messages.create(
                model=MODELO_CLAUDE, max_tokens=600, system=_instrucciones(),
                tools=ESQUEMAS, messages=mensajes)
            if resp.stop_reason != "tool_use":
                texto = "".join(b.text for b in resp.content if b.type == "text").strip()
                self.historial_chat.append({"role": "assistant", "content": texto})
                self.historial_chat = self.historial_chat[-10:]
                return texto
            mensajes.append({"role": "assistant", "content": resp.content})
            resultados = []
            for b in resp.content:
                if b.type == "tool_use":
                    try:
                        salida = HERRAMIENTAS[b.name](**b.input)
                    except Exception as e:
                        salida = {"error": str(e)}
                    resultados.append({"type": "tool_result", "tool_use_id": b.id,
                                       "content": json.dumps(salida, ensure_ascii=False,
                                                             default=str)})
            mensajes.append({"role": "user", "content": resultados})
        return "No pude completar la consulta. Intenta preguntarlo de otra forma."


# ----------------------------------------------------------------------
# Modo de respaldo: reglas simples sin internet
# ----------------------------------------------------------------------
_NOMBRES_DIAS = {"lunes": 0, "martes": 1, "miercoles": 2, "miércoles": 2,
                 "jueves": 3, "viernes": 4, "sabado": 5, "sábado": 5, "domingo": 6}
_PATRON_PLACA = re.compile(r"\b([A-Za-z]{3})\s*-?\s*(\d{2}[A-Za-z0-9])\b")


def _fecha_de_pregunta(q):
    hoy = date.today()
    if "pasado mañana" in q or "pasado manana" in q:
        return hoy + timedelta(days=2)
    if "mañana" in q or "manana" in q:
        return hoy + timedelta(days=1)
    m = re.search(r"(\d{1,2})/(\d{1,2})(?:/(\d{4}))?", q)
    if m:
        return date(int(m.group(3) or hoy.year), int(m.group(2)), int(m.group(1)))
    for nombre, wd in _NOMBRES_DIAS.items():
        if re.search(rf"\b{nombre}\b", q):
            return hoy + timedelta(days=(wd - hoy.weekday()) % 7)
    return hoy


def _hora_de_pregunta(q):
    m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\. ?m\.|p\. ?m\.)?", q)
    if not m or not re.search(r"\b(a las|hora|am|pm|a\. ?m|p\. ?m)", q):
        return None
    h, mi = int(m.group(1)), int(m.group(2) or 0)
    if m.group(3) and m.group(3).startswith("p") and h < 12:
        h += 12
    return f"{h:02d}:{mi:02d}" if 0 <= h < 24 else None


def responder_sin_llm(pregunta):
    q = pregunta.lower()
    m = _PATRON_PLACA.search(pregunta)
    if m:
        placa = (m.group(1) + m.group(2)).upper()
        q = q.replace(m.group(0).lower(), " ")          # que la placa no se lea como hora
        d = _fecha_de_pregunta(q)
        r = herramienta_verificar_placa(placa, d.isoformat(), _hora_de_pregunta(q))
        if not r.get("valida"):
            return r["explicacion"]
        prox = r["proximos_dias_con_restriccion"]
        extra = f" Próximos días con restricción: {', '.join(prox)}." if prox else ""
        return r["explicacion"] + extra
    if re.search(r"detect|cámara|camara|historial|vist|pasaron|registr", q):
        filas = leer_historial(5)
        if not filas:
            return "Todavía no hay placas detectadas por la cámara."
        lineas = [f"{f['fecha_hora'][11:16]} {f['placa']} "
                  f"({'con restricción' if f['restringido'] == 'si' else 'sin restricción'})"
                  for f in filas]
        return "Últimas placas detectadas: " + "; ".join(lineas) + "."
    if re.search(r"multa|sanci|comparendo|inmoviliz", q):
        return pyp.CONFIG["sancion"] + " " + pyp.CONFIG["excepciones"]
    if re.search(r"excep|discapacid|exent", q):
        return pyp.CONFIG["excepciones"]
    if re.search(r"horario|qué hora|que hora|hasta qué|hasta que|desde qué", q):
        return (f"El pico y placa en Popayán rige de {pyp.CONFIG['hora_inicio']} a "
                f"{pyp.CONFIG['hora_fin']}, de lunes a viernes. {pyp.CONFIG['nota_horario']}")
    if re.search(r"qu[eé] placas|d[ií]gitos|cu[aá]les|pico y placa (hoy|mañana|el)|"
                 r"tiene[n]? pico", q):
        d = _fecha_de_pregunta(q)
        dig = pyp.digitos_del_dia(d)
        if not dig:
            return f"El {pyp.fecha_texto(d)} no hay pico y placa: {pyp.motivo_sin_medida(d)}."
        return (f"El {pyp.fecha_texto(d)} tienen pico y placa las placas terminadas en "
                f"{dig[0]} y {dig[1]}, de {pyp.CONFIG['hora_inicio']} a {pyp.CONFIG['hora_fin']}.")
    if re.search(r"regla|c[oó]mo funciona|rotaci|explica", q):
        return pyp.resumen_reglas()
    return ("Puedo ayudarte con el pico y placa de Popayán. Pregúntame, por ejemplo: "
            "'¿La placa ABC123 puede salir hoy?', '¿Qué placas tienen pico y placa el "
            "viernes?' o '¿Qué placas detectó la cámara?'.")


# %% CHAT EN LA CONSOLA DE SPYDER
if __name__ == "__main__":
    a = Asistente()
    print(f"Asistente de pico y placa - modo {a.modo}. Escriba 'salir' para terminar.")
    while True:
        try:
            q = input("\nUsted: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"salir", "exit", "q"}:
            break
        if q:
            print("Asistente:", a.preguntar(q))

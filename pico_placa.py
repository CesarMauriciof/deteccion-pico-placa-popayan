# -*- coding: utf-8 -*-
"""
pico_placa.py - Reglas de pico y placa (Popayán, configurable)
==============================================================
Las reglas NO están escritas en el código: se leen de config_popayan.json.
Si la Alcaldía cambia la rotación, basta con editar ese archivo.

Funciones principales:
    verificar(placa, fecha_hora)   -> ¿puede circular?, con explicación
    digitos_del_dia(fecha)         -> qué dígitos tienen restricción ese día
    proximos_dias(placa)           -> próximos días con restricción
"""

import os
import re
import json
from datetime import datetime, date, time, timedelta

try:
    import holidays                      # festivos de Colombia
    FESTIVOS = holidays.Colombia(years=range(2024, 2031))
except ImportError:                      # si no está instalado, sin festivos
    FESTIVOS = {}

RUTA_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "config_popayan.json")
DIAS = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
DIAS_TILDE = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

PATRON_CARRO = re.compile(r"^[A-Z]{3}\d{3}$")     # ABC123
PATRON_MOTO = re.compile(r"^[A-Z]{3}\d{2}[A-Z]$")  # ABC12D


def cargar_config(ruta=RUTA_CONFIG):
    with open(ruta, encoding="utf-8") as f:
        cfg = json.load(f)
    h1, m1 = map(int, cfg["hora_inicio"].split(":"))
    h2, m2 = map(int, cfg["hora_fin"].split(":"))
    cfg["_inicio"], cfg["_fin"] = time(h1, m1), time(h2, m2)
    return cfg


CONFIG = cargar_config()


def fecha_texto(d):
    return f"{DIAS_TILDE[d.weekday()]} {d.day} de {MESES[d.month - 1]} de {d.year}"


def normalizar_placa(texto):
    """Quita espacios y guiones; devuelve (placa, tipo) o (None, None)."""
    p = re.sub(r"[^A-Za-z0-9]", "", str(texto)).upper()
    if PATRON_CARRO.match(p):
        return p, "carro"
    if PATRON_MOTO.match(p):
        return p, "moto"
    return None, None


def ultimo_digito(placa):
    return int([c for c in placa if c.isdigit()][-1])


def motivo_sin_medida(d, cfg=CONFIG):
    """Explica por qué no hay pico y placa ese día (o None si sí hay)."""
    if d in FESTIVOS and not cfg["aplica_festivos"]:
        return f"es festivo ({FESTIVOS.get(d)})"
    if d.weekday() == 5 and not cfg["aplica_sabados"]:
        return "los sábados no aplica"
    if d.weekday() == 6 and not cfg["aplica_domingos"]:
        return "los domingos no aplica"
    return None


def digitos_del_dia(d, cfg=CONFIG):
    """Dígitos con restricción en la fecha d (lista vacía si no aplica)."""
    if isinstance(d, datetime):
        d = d.date()
    if motivo_sin_medida(d, cfg):
        return []
    return cfg["rotacion"].get(DIAS[d.weekday()], [])


def verificar(texto_placa, fecha_hora=None, cfg=CONFIG):
    """¿La placa tiene pico y placa en esa fecha y hora?

    Devuelve un diccionario con 'restringido' (True/False) y 'explicacion'.
    """
    fecha_hora = fecha_hora or datetime.now()
    placa, tipo = normalizar_placa(texto_placa)
    if placa is None:
        return {"valida": False, "placa": texto_placa,
                "explicacion": f"'{texto_placa}' no tiene formato de placa "
                               "colombiana (ABC123 para carro, ABC12D para moto)."}
    d = fecha_hora.date()
    digito = ultimo_digito(placa)
    digitos = digitos_del_dia(d, cfg)
    en_horario = cfg["_inicio"] <= fecha_hora.time() < cfg["_fin"]
    dia_con_pyp = digito in digitos
    restringido = dia_con_pyp and en_horario

    horario = f"de {cfg['hora_inicio']} a {cfg['hora_fin']}"
    motivo = motivo_sin_medida(d, cfg)
    if motivo:
        expl = f"La placa {placa} puede circular: el {fecha_texto(d)} {motivo}."
    elif not dia_con_pyp:
        expl = (f"La placa {placa} (termina en {digito}) puede circular el "
                f"{fecha_texto(d)}. Ese día la restricción es para placas "
                f"terminadas en {digitos[0]} y {digitos[1]}.")
    elif restringido:
        expl = (f"La placa {placa} (termina en {digito}) TIENE pico y placa el "
                f"{fecha_texto(d)} {horario}: a las {fecha_hora:%H:%M} no puede circular.")
    else:
        expl = (f"La placa {placa} (termina en {digito}) tiene pico y placa el "
                f"{fecha_texto(d)} {horario}, pero a las {fecha_hora:%H:%M} está "
                "fuera del horario de restricción.")
    return {"valida": True, "placa": placa, "tipo": tipo, "ultimo_digito": digito,
            "fecha": d.isoformat(), "dia": DIAS_TILDE[d.weekday()],
            "digitos_restringidos_dia": digitos, "dia_con_pico_y_placa": dia_con_pyp,
            "en_horario": en_horario, "restringido": restringido,
            "horario": horario, "explicacion": expl}


def proximos_dias(texto_placa, desde=None, n=3, cfg=CONFIG):
    """Próximas n fechas en que la placa tiene pico y placa."""
    placa, _ = normalizar_placa(texto_placa)
    if placa is None:
        return []
    d = (desde or date.today())
    if isinstance(d, datetime):
        d = d.date()
    digito, fechas = ultimo_digito(placa), []
    for i in range(0, 60):
        dia = d + timedelta(days=i)
        if digito in digitos_del_dia(dia, cfg):
            fechas.append(dia)
            if len(fechas) == n:
                break
    return fechas


def resumen_reglas(cfg=CONFIG):
    filas = [f"  {DIAS_TILDE[DIAS.index(k)].capitalize():<10} placas terminadas en "
             f"{v[0]} y {v[1]}" for k, v in cfg["rotacion"].items()]
    return (f"Pico y placa en {cfg['ciudad']} ({cfg['vigencia']}).\n"
            f"Aplica a: {cfg['vehiculos']}.\n"
            f"Horario: {cfg['hora_inicio']} a {cfg['hora_fin']}, de lunes a viernes; "
            "no aplica sábados, domingos ni festivos.\n" + "\n".join(filas) +
            f"\nSanción: {cfg['sancion']}\nNota: {cfg['nota_horario']}")


if __name__ == "__main__":
    print(resumen_reglas())
    for p in ["JMK125", "ABC120", "XYZ12D"]:
        print(verificar(p, datetime(2026, 11, 17, 9, 30))["explicacion"])

"""
Descarga la tasa de desempleo mensual por estado (BLS, Local Area Unemployment
Statistics) y la guarda en `datos_externos/desempleo_estatal_bls.csv`.

El CSV ya está en el repositorio: este script solo hace falta para regenerarlo
o extender el período.

Fuente: archivo `la.data.3.AllStatesS` de https://download.bls.gov/pub/time.series/la/
- Series LASST{FIPS}0000000000003: tasa de desempleo (%), desestacionalizada.
- 50 estados + DC + Puerto Rico, enero 2006 a diciembre 2016.

El BLS rechaza pedidos automáticos sin un User-Agent con un contacto, así que
el script se identifica con un mail (cambiarlo por el propio).

Uso:
    python scripts/download_desempleo.py
"""
import io
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://download.bls.gov/pub/time.series/la/la.data.3.AllStatesS"
CONTACTO = "tu_mail@ejemplo.com"  # reemplazar por un mail real
DESTINO = Path(__file__).resolve().parents[1] / "datos_externos" / "desempleo_estatal_bls.csv"
DESDE, HASTA = 2006, 2016

FIPS_A_ESTADO = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT",
    "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL",
    "18": "IN", "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD",
    "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT", "31": "NE",
    "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA", "54": "WV",
    "55": "WI", "56": "WY", "72": "PR",
}


def main():
    pedido = urllib.request.Request(URL, headers={"User-Agent": f"ITBA TP ({CONTACTO})"})
    with urllib.request.urlopen(pedido) as respuesta:
        texto = respuesta.read().decode("utf-8")

    df = pd.read_csv(io.StringIO(texto), sep="\t", dtype=str)
    df.columns = df.columns.str.strip()
    df = df.apply(lambda s: s.str.strip())
    es_tasa = df["series_id"].str.match(r"LASST\d{2}0000000000003$")
    mensual = df["period"].str.match(r"M(0[1-9]|1[0-2])$")
    df = df[es_tasa & mensual]
    df["year"] = df["year"].astype(int)
    df = df[df["year"].between(DESDE, HASTA)]

    salida = pd.DataFrame({
        "estado": df["series_id"].str[5:7].map(FIPS_A_ESTADO),
        "mes": df["year"].astype(str) + "-" + df["period"].str[1:],
        "tasa_desempleo": df["value"].astype(float),
    }).sort_values(["estado", "mes"])
    DESTINO.parent.mkdir(exist_ok=True)
    salida.to_csv(DESTINO, index=False)
    print(f"Listo: {DESTINO} ({len(salida):,} filas, {salida['estado'].nunique()} estados)")


if __name__ == "__main__":
    main()

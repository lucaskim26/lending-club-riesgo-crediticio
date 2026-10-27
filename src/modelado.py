"""
Modelos de PD de la Entrega 03: variables macro, preprocesamiento, validación
temporal y guardado/carga del modelo final.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import features as F

RAIZ = Path(__file__).resolve().parents[1]
RUTA_DESEMPLEO = RAIZ / "datos_externos" / "desempleo_estatal_bls.csv"
CARPETA_MODELOS = RAIZ / "modelos"

# ---------------------------------------------------------------------------
# Variables macroeconómicas (desempleo por estado, BLS LAUS)
# ---------------------------------------------------------------------------

MACRO = ["desempleo_estado", "desempleo_cambio_12m"]


def agregar_macro(df, rezago_meses=1):
    """Agrega el desempleo del estado del solicitante, sin leakage.

    - desempleo_estado: tasa de desempleo (%) del estado `rezago_meses` antes
      del mes de originación (por defecto, el mes anterior).
    - desempleo_cambio_12m: cuánto cambió esa tasa en los 12 meses previos
      (puntos porcentuales). Positivo = la economía del estado empeora.

    `df` necesita `addr_state` y `fecha_originacion`. Devuelve un DataFrame nuevo.
    """
    tasa = pd.read_csv(RUTA_DESEMPLEO)
    tasa["mes"] = pd.PeriodIndex(tasa["mes"], freq="M")
    tasa = tasa.set_index(["estado", "mes"])["tasa_desempleo"]

    mes_dato = df["fecha_originacion"].dt.to_period("M") - rezago_meses
    def buscar(meses):
        claves = pd.MultiIndex.from_arrays([df["addr_state"], meses])
        return tasa.reindex(claves).to_numpy()

    out = df.copy()
    out["desempleo_estado"] = buscar(mes_dato)
    out["desempleo_cambio_12m"] = out["desempleo_estado"] - buscar(mes_dato - 12)
    return out


# ---------------------------------------------------------------------------
# Preprocesamiento y modelos
# ---------------------------------------------------------------------------

def columnas_modelo(con_macro=True, con_estado=True):
    """Listas de columnas (numéricas, binarias, categóricas) de cada variante."""
    numericas, binarias, categoricas = F.features_modelo()
    if con_macro:
        numericas = numericas + MACRO
    if not con_estado:
        categoricas = [c for c in categoricas if c != "addr_state"]
    return numericas, binarias, categoricas


def crear_preprocesador_arboles(numericas, binarias, categoricas):
    """Para los árboles: numéricas y binarias sin cambios (los vacíos los
    manejan solos) y categóricas en columnas 0/1, igual que la logística."""
    codificador = OneHotEncoder(
        handle_unknown="infrequent_if_exist", min_frequency=0.005, sparse_output=False)
    return ColumnTransformer([
        ("num", "passthrough", numericas + binarias),
        ("cat", codificador, categoricas),
    ], verbose_feature_names_out=False).set_output(transform="pandas")


def crear_modelo(tipo, params=None, con_macro=True, con_estado=True):
    """Pipeline completo (preprocesamiento + modelo) listo para `fit`."""
    params = params or {}
    numericas, binarias, categoricas = columnas_modelo(con_macro, con_estado)
    if tipo == "logistica":
        prepro = F.crear_preprocesador(numericas, binarias, categoricas)
        modelo = LogisticRegression(max_iter=2000, **params)
    elif tipo == "random_forest":
        prepro = crear_preprocesador_arboles(numericas, binarias, categoricas)
        base = dict(n_estimators=200, min_samples_leaf=100, max_features="sqrt",
                    max_samples=0.3, n_jobs=-1, random_state=42)
        modelo = RandomForestClassifier(**{**base, **params})
    elif tipo == "lightgbm":
        prepro = crear_preprocesador_arboles(numericas, binarias, categoricas)
        base = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
                    min_child_samples=200, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1,
                    random_state=42, verbose=-1)
        modelo = LGBMClassifier(**{**base, **params})
    else:
        raise ValueError(f"Tipo de modelo desconocido: {tipo}")
    return Pipeline([("preprocesamiento", prepro), ("modelo", modelo)])


# ---------------------------------------------------------------------------
# Validación temporal dentro de train
# ---------------------------------------------------------------------------

def cortes_temporales(fechas, anios_validacion):
    """Para cada año de validación: entrenar con todo lo anterior y validar
    con ese año (ventana creciente). Devuelve (año, índices_train, índices_val)."""
    anio = fechas.dt.year.to_numpy()
    for a in anios_validacion:
        yield a, np.flatnonzero(anio < a), np.flatnonzero(anio == a)


# ---------------------------------------------------------------------------
# Guardar y cargar el modelo final
# ---------------------------------------------------------------------------

def guardar_modelo(pipeline, metadatos, nombre="modelo_pd"):
    """Guarda el pipeline (.joblib) y sus metadatos (.json) en `modelos/`."""
    CARPETA_MODELOS.mkdir(exist_ok=True)
    ruta = CARPETA_MODELOS / f"{nombre}.joblib"
    joblib.dump(pipeline, ruta, compress=3)
    with open(CARPETA_MODELOS / f"{nombre}.json", "w", encoding="utf-8") as archivo:
        json.dump(metadatos, archivo, ensure_ascii=False, indent=2, default=float)
    return ruta


def cargar_modelo(nombre="modelo_pd"):
    """Devuelve (pipeline, metadatos) sin reentrenar."""
    pipeline = joblib.load(CARPETA_MODELOS / f"{nombre}.joblib")
    with open(CARPETA_MODELOS / f"{nombre}.json", encoding="utf-8") as archivo:
        metadatos = json.load(archivo)
    return pipeline, metadatos


def predecir_pd(pipeline, df):
    """PD de cada préstamo. `df` debe tener las columnas del pipeline
    (variables de `features_modelo` + macro si el modelo las usa)."""
    return pipeline.predict_proba(df)[:, 1]

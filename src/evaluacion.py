"""
Métricas y tablas para evaluar probabilidades de default (PD).

Todas las funciones reciben predicciones ya calculadas: no entrenan nada.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)


def ks(y, score):
    """Estadístico de Kolmogorov-Smirnov: máxima distancia entre la
    distribución del score de los que pagaron y la de los que no pagaron."""
    fpr, tpr, _ = roc_curve(y, score)
    return float(np.max(tpr - fpr))


def metricas_pd(y, pd_estimada):
    """Métricas que no dependen de un umbral.

    - roc_auc: probabilidad de que un default tenga más PD que un pagado.
    - pr_auc: precisión promedio al ir bajando el umbral (average precision).
    - ks: máxima separación entre pagados y defaults.
    - brier: error cuadrático medio de la probabilidad (más bajo = mejor).
    - pd_media / default_observado: si coinciden, el nivel está bien calibrado.
    """
    y = np.asarray(y)
    p = np.asarray(pd_estimada, dtype=float)
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "ks": ks(y, p),
        "brier": brier_score_loss(y, p),
        "pd_media": p.mean(),
        "default_observado": y.mean(),
    }


def tabla_calibracion(y, pd_estimada, grupos=10):
    """PD promedio y default observado por decil de PD."""
    datos = pd.DataFrame({"pd": np.asarray(pd_estimada, dtype=float), "default": np.asarray(y)})
    datos["grupo"] = pd.qcut(datos["pd"].rank(method="first"), grupos, labels=False) + 1
    return datos.groupby("grupo").agg(
        prestamos=("pd", "size"), pd_media=("pd", "mean"), default_observado=("default", "mean"))


def asignar_banda(pd_estimada, cortes, nombres):
    """Banda de riesgo de cada préstamo. `cortes` son los límites internos
    (crecientes); una PD igual a un corte cae en la banda de la derecha."""
    indice = np.searchsorted(np.asarray(cortes), np.asarray(pd_estimada, dtype=float), side="right")
    return pd.Categorical(np.asarray(nombres)[indice], categories=nombres, ordered=True)


def tabla_por_banda(banda, y, pd_estimada, perdida_sobre_monto=None, severidad=None):
    """Resumen por banda: cantidad, PD media, default observado y, si se pasan,
    la pérdida esperada (PD × severidad) y la pérdida observada sobre el monto.
    La pérdida de los pagados es 0; un default con pérdida faltante es un error,
    no una pérdida nula. Las pérdidas y PD se expresan como fracciones [0, 1].

    Las entradas son vectores de igual longitud y orden. Si varias son Series,
    sus índices deben coincidir: no se realinean ni se mezclan silenciosamente.
    """
    entradas = {"banda": banda, "y": y, "pd_estimada": pd_estimada}
    if perdida_sobre_monto is not None:
        entradas["perdida_sobre_monto"] = perdida_sobre_monto
    if any(np.ndim(v) != 1 for v in entradas.values()):
        raise ValueError("Las entradas deben ser vectores de una dimensión.")
    if not len(y) or any(len(v) != len(y) for v in entradas.values()):
        raise ValueError("Las entradas deben tener igual longitud y no estar vacías.")
    indices = [v.index for v in entradas.values() if isinstance(v, pd.Series)]
    if any(not indice.equals(indices[0]) for indice in indices[1:]):
        raise ValueError("Los índices de las Series deben coincidir en el mismo orden.")

    objetivo = pd.Series(y).to_numpy(dtype=float, na_value=np.nan)
    probabilidad = pd.Series(pd_estimada).to_numpy(dtype=float, na_value=np.nan)
    if not np.isin(objetivo, [0, 1]).all():
        raise ValueError("y debe contener solo 0 (pagado) y 1 (default), sin faltantes.")
    if not np.isfinite(probabilidad).all() or ((probabilidad < 0) | (probabilidad > 1)).any():
        raise ValueError("La PD debe ser finita y estar entre 0 y 1.")
    if pd.isna(banda).any():
        raise ValueError("Cada préstamo debe tener una banda, sin faltantes.")
    if severidad is not None:
        if np.ndim(severidad) != 0 or not np.isfinite(severidad) or not 0 <= severidad <= 1:
            raise ValueError("La severidad debe ser un escalar finito entre 0 y 1.")

    # .array conserva las categorías y evita la alineación automática por índice.
    datos = pd.DataFrame({"banda": pd.Series(banda).array, "pd": probabilidad,
                          "default": objetivo})
    agregados = {"prestamos": ("pd", "size"), "pd_media": ("pd", "mean"),
                 "default_observado": ("default", "mean")}
    if perdida_sobre_monto is not None:
        perdida = pd.Series(perdida_sobre_monto).to_numpy(dtype=float, na_value=np.nan).copy()
        es_default = objetivo == 1
        if np.isnan(perdida[es_default]).any():
            raise ValueError("Hay defaults con pérdida faltante; completar o revisar esos datos.")
        informada = ~np.isnan(perdida)
        if not np.isfinite(perdida[informada]).all() or (
                (perdida[informada] < 0) | (perdida[informada] > 1)).any():
            raise ValueError("La pérdida informada debe ser finita y estar entre 0 y 1.")
        if (perdida[~es_default & informada] != 0).any():
            raise ValueError("La pérdida informada de un préstamo pagado debe ser 0.")
        perdida[~es_default] = 0.0
        datos["perdida"] = perdida
        agregados["perdida_observada"] = ("perdida", "mean")
    tabla = datos.groupby("banda", observed=False).agg(**agregados)
    tabla.insert(1, "% de prestamos", tabla["prestamos"] / tabla["prestamos"].sum())
    if severidad is not None:
        tabla["perdida_esperada"] = tabla["pd_media"] * severidad
    return tabla

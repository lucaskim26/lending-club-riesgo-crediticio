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
    La pérdida de los préstamos pagados es 0."""
    datos = pd.DataFrame({"banda": banda, "pd": np.asarray(pd_estimada, dtype=float),
                          "default": np.asarray(y)})
    agregados = {"prestamos": ("pd", "size"), "pd_media": ("pd", "mean"),
                 "default_observado": ("default", "mean")}
    if perdida_sobre_monto is not None:
        datos["perdida"] = np.nan_to_num(np.asarray(perdida_sobre_monto, dtype=float), nan=0.0)
        agregados["perdida_observada"] = ("perdida", "mean")
    tabla = datos.groupby("banda", observed=False).agg(**agregados)
    tabla.insert(1, "% de prestamos", tabla["prestamos"] / tabla["prestamos"].sum())
    if severidad is not None:
        tabla["perdida_esperada"] = tabla["pd_media"] * severidad
    return tabla

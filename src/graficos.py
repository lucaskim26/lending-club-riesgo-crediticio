"""Estilo común para los gráficos del proyecto."""
from pathlib import Path

import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[1]
CARPETA_FIGURAS = RAIZ / "figures"

# Paleta categórica validada para daltonismo (orden fijo, no se cicla)
AZUL = "#2a78d6"      # serie principal / Fully Paid / train
NARANJA = "#eb6834"   # serie secundaria / Charged Off / test
AQUA = "#1baf7a"
GRIS = "#8a8984"      # referencias y anotaciones
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
CMAP_SECUENCIAL = "Blues"


def aplicar_estilo():
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelcolor": TINTA_SECUNDARIA,
        "axes.edgecolor": "#d6d5d0",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": "#ebeae6",
        "grid.linewidth": 0.8,
        "xtick.color": TINTA_SECUNDARIA,
        "ytick.color": TINTA_SECUNDARIA,
        "legend.frameon": False,
        "lines.linewidth": 2,
    })


def guardar(fig, nombre):
    CARPETA_FIGURAS.mkdir(exist_ok=True)
    fig.savefig(CARPETA_FIGURAS / f"{nombre}.png")

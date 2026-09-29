"""
Descarga el dataset de préstamos de Lending Club usado en este TP.

Fuente: Kaggle, "Lending Club Loan Data" (wordsforthewise/lending-club).
Se descarga solo el archivo de préstamos otorgados 2007-2018
(`accepted_2007_to_2018Q4.csv.gz`, ~393 MB, 2,26M préstamos x 151 columnas).

Requiere un token de la API de Kaggle (kaggle.com -> Settings -> API ->
Create New Token), guardado según las instrucciones que muestra Kaggle
(por ejemplo, en ~/.kaggle/access_token).

Uso:
    python scripts/download_data.py
"""
import subprocess
from pathlib import Path

DATASET = "wordsforthewise/lending-club"
ARCHIVO = "accepted_2007_to_2018Q4.csv.gz"
DESTINO = Path(__file__).resolve().parents[1] / "data"


def main():
    DESTINO.mkdir(exist_ok=True)
    if (DESTINO / ARCHIVO).exists():
        print(f"Ya existe: {DESTINO / ARCHIVO}")
        return
    print(f"Descargando {ARCHIVO} desde kaggle.com/datasets/{DATASET} ...")
    subprocess.run(
        ["kaggle", "datasets", "download", DATASET, "-f", ARCHIVO, "-p", str(DESTINO)],
        check=True,
    )
    size_mb = (DESTINO / ARCHIVO).stat().st_size / (1024 * 1024)
    print(f"Listo: {DESTINO / ARCHIVO} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()

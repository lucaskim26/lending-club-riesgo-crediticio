# Modelo de riesgo de default crediticio — Lending Club

**Ciencia de Datos Aplicada · ITBA**
Gianluca Giannine Lizarraga · Timoteo Harrington · Lucas Kim

Queremos estimar la probabilidad de que una persona no devuelva un préstamo (PD), usando solo lo que se sabe cuando lo pide. Con esa probabilidad armamos bandas de riesgo y estimamos cuánto se perdería. Usamos los datos históricos de Lending Club, una plataforma de préstamos entre personas de Estados Unidos.

## Estado del proyecto

| Entrega | Contenido | Estado |
|---|---|---|
| E01 — Propuesta | Problema, objetivos y alcance | ✅ Entregado |
| E02 — Recopilación y preparación de datos | Datos, calidad, leakage, separación train/test, exploración, pérdida y transformaciones | ✅ Entregado |
| E03 — Modelado | Regresión logística, Random Forest y LightGBM, validación por año, evaluación, bandas y modelo guardado | ✅ Este repo |
| E04 — Despliegue y presentación | Prima de riesgo e informe final | Pendiente |

## Qué hay en cada carpeta

```
├── data/            # Datos (no se suben a GitHub; se descargan con el script)
├── datos_externos/  # Desempleo mensual por estado (BLS), sí está en el repo
├── figures/         # Gráficos que generan los notebooks
├── modelos/         # Modelo final guardado (Entrega 03)
├── notebooks/
│   ├── 02_recopilacion_preparacion_datos.ipynb   # Entrega 02
│   └── 03_modelado.ipynb                         # Entrega 03
├── scripts/
│   ├── download_data.py         # Descarga el dataset de Kaggle
│   └── download_desempleo.py    # Vuelve a bajar el desempleo del BLS
├── src/
│   ├── datos.py       # Carga los datos y se queda con los préstamos terminados
│   ├── features.py    # Clasifica las columnas, crea variables y calcula la pérdida
│   ├── modelado.py    # Desempleo, modelos, validación por año, guardar/cargar el modelo
│   ├── evaluacion.py  # Métricas (AUC, KS, calibración) y tablas por banda
│   └── graficos.py    # Colores y estilo de los gráficos
├── requirements.txt
└── README.md
```

## Cómo verlo

- **Sin instalar nada:** abrí los notebooks acá en GitHub. Están guardados con todas sus tablas y gráficos.
- **En Google Colab:** abrí el notebook en Colab y ejecutá todo. La primera celda baja este repositorio y te pide un token de Kaggle para descargar los datos (en kaggle.com → Settings → API). El notebook 03 necesita los archivos que genera el 02; si no están, su primera celda ejecuta el 02 sola (~10 minutos más). El 03 tarda ~15 minutos.
- **En tu computadora:**

```bash
git clone https://github.com/lucaskim26/lending-club-riesgo-crediticio
cd lending-club-riesgo-crediticio
pip install -r requirements.txt
python scripts/download_data.py      # necesita el token de Kaggle
jupyter notebook notebooks/02_recopilacion_preparacion_datos.ipynb   # después, 03_modelado.ipynb
```

La primera vez tarda unos minutos y usa ~5 GB de memoria, porque lee el archivo completo. Después guarda una copia en `data/` y las siguientes veces es rápido.

## Datos

[*Lending Club Loan Data*](https://www.kaggle.com/datasets/wordsforthewise/lending-club) en Kaggle: todos los préstamos que dio Lending Club entre 2007 y 2018 (2,26 millones de préstamos, 151 columnas).

## Cómo lo hicimos (resumen)

1. **Qué predecimos:** si el préstamo se dio por perdido (`Charged Off` = 1) o se pagó (`Fully Paid` = 0).
2. **Qué préstamos usamos:** solo los que ya terminaron, a 36 meses, de 2007 a 2015 (618 mil). En los años recientes muchos préstamos seguían abiertos y el default aparecía inflado.
3. **Qué columnas usamos:** solo las que existen al pedir el préstamo. Sacamos las que aparecen después (leakage) y las que son decisiones de Lending Club, como su nota (`grade`) o si verificó el ingreso.
4. **Train y test:** separados por fecha (train hasta agosto de 2015, test de septiembre a diciembre de 2015), antes de explorar los datos.
5. **Pérdida:** cuando alguien no paga se pierde ~89% de lo que debía, que es el 51,7% del monto prestado.
6. **Transformaciones:** variables nuevas, recorte de valores extremos, relleno de vacíos, misma escala y categorías pasadas a columnas 0/1. Todo lo que aprende de los datos se ajusta solo con train.
7. **Resultado final del sistema:** una pérdida esperada por persona, PD × 51,7%. Es la pérdida de toda la vida del préstamo, no una tasa de interés.
8. **Modelos (E03):** regresión logística, Random Forest y LightGBM, elegidos con validación por año dentro de train (2013, 2014 y 2015). Ganó **LightGBM**: en test tiene ROC-AUC 0,704, contra 0,683 de la nota de Lending Club en los mismos préstamos.
9. **Desempleo por estado (BLS):** su aporte es despreciable: quitarlo reduce el AUC de validación de 0,6837 a 0,6836. Quedó afuera para simplificar, no porque perjudique el modelo.
10. **Bandas:** 7 bandas con cortes definidos en train; en test el default sube de 4% (banda 1) a 57% (banda 7).
11. **Modelo guardado:** `modelos/modelo_pd.joblib` (se carga con `src.modelado.cargar_modelo()`) y `modelos/modelo_pd.json` con los cortes, la severidad y las métricas.

## Verificación sin entrenar

El modelo guardado recibe **54 variables ya construidas** y las transforma en **102 columnas** para LightGBM. Se puede probar la carga e inferencia con cinco casos sintéticos sin descargar el dataset ni ejecutar los notebooks:

```bash
python -m pip install -r requirements-inference.txt
python -m unittest discover -s tests -v
```

Usá un entorno aislado. La [guía de verificación de E03](docs/verificacion_e03.md) detalla las versiones originales y verificadas, el contrato de entrada, las pruebas y sus límites.

**Alcance:** la evaluación es retrospectiva, con separación por fecha de originación; no reconstruye qué etiquetas estaban disponibles en cada fecha histórica. La selección de variables de E02 se hizo sobre todo train antes de los folds, aunque los parámetros del pipeline sí se ajustan dentro de cada corte. Test se mantuvo separado. Estas aclaraciones y la validación de pérdidas faltantes no reentrenan el modelo ni cambian los resultados guardados.

## Autores

Gianluca Giannine Lizarraga · Timoteo Harrington · Lucas Kim — ITBA, 2026

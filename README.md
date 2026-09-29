# Modelo de riesgo de default crediticio — Lending Club

**Ciencia de Datos Aplicada · ITBA**
Gianluca Giannine Lizarraga · Timoteo Harrington · Lucas Kim

Queremos estimar la probabilidad de que una persona no devuelva un préstamo (PD), usando solo lo que se sabe cuando lo pide. Con esa probabilidad armamos bandas de riesgo y estimamos cuánto se perdería. Usamos los datos históricos de Lending Club, una plataforma de préstamos entre personas de Estados Unidos.

## Estado del proyecto

| Entrega | Contenido | Estado |
|---|---|---|
| E01 — Propuesta | Problema, objetivos y alcance | ✅ Entregado |
| E02 — Recopilación y preparación de datos | Datos, calidad, leakage, separación train/test, exploración, pérdida y transformaciones | ✅ Este repo |
| E03 — Modelado | Regresión logística, Random Forest, LightGBM/XGBoost, evaluación y bandas | Pendiente |
| E04 — Despliegue y presentación | Prima de riesgo e informe final | Pendiente |

## Qué hay en cada carpeta

```
├── data/          # Datos (no se suben a GitHub; se descargan con el script)
├── figures/       # Gráficos que genera el notebook
├── notebooks/
│   └── 02_recopilacion_preparacion_datos.ipynb   # Entrega 02
├── scripts/
│   └── download_data.py   # Descarga el dataset de Kaggle
├── src/
│   ├── datos.py       # Carga los datos y se queda con los préstamos terminados
│   ├── features.py    # Clasifica las columnas, crea variables y calcula la pérdida
│   └── graficos.py    # Colores y estilo de los gráficos
├── requirements.txt
└── README.md
```

## Cómo verlo

- **Sin instalar nada:** abrí el notebook acá en GitHub. Está guardado con todas sus tablas y gráficos.
- **En Google Colab:** abrí el notebook en Colab y ejecutá todo. La primera celda baja este repositorio y te pide un token de Kaggle para descargar los datos (en kaggle.com → Settings → API).
- **En tu computadora:**

```bash
git clone https://github.com/lucaskim26/lending-club-riesgo-crediticio
cd lending-club-riesgo-crediticio
pip install -r requirements.txt
python scripts/download_data.py      # necesita el token de Kaggle
jupyter notebook notebooks/02_recopilacion_preparacion_datos.ipynb
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

## Autores

Gianluca Giannine Lizarraga · Timoteo Harrington · Lucas Kim — ITBA, 2026

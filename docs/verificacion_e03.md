# Verificación de E03 sin reentrenar

## Alcance metodológico

La separación train/test y los cortes internos se hacen por fecha de **originación**. Es una evaluación retrospectiva sobre préstamos resueltos, no una reconstrucción de la información disponible al comenzar cada año. Las etiquetas finales pueden conocerse años después. Un backtest operativo requiere respetar esa disponibilidad y la maduración o censura de los préstamos.

E02 eligió variables redundantes con correlaciones y AUC univariado sobre todo train. E03 ajusta los parámetros del pipeline dentro de cada corte, pero no repite esa selección: la validación interna puede ser optimista. Test se mantuvo separado de la selección. Incorporar la selección dentro de cada fold requiere una nueva ejecución y no forma parte de esta corrección.

Quitar el desempleo reduce el AUC promedio de validación de 0,6837 a 0,6836. La elección del modelo sin desempleo se justifica por su aporte despreciable y la simplificación de la inferencia, no porque perjudique el desempeño.

## Contrato de entrada y persistencia

- `modelos/modelo_pd.joblib` contiene el pipeline ajustado; `src.modelado.cargar_modelo()` devuelve ese pipeline y los metadatos de `modelo_pd.json`.
- La inferencia recibe las **54 variables ya construidas** enumeradas en el JSON: 47 numéricas, 4 binarias y 3 categóricas. No recibe directamente las 151 columnas crudas. La construcción fila a fila está en `src.features.construir_features`.
- El LightGBM guardado recibe **102 columnas transformadas**. Las 105 de E02 corresponden al preprocesamiento de logística, con indicadores de faltantes.
- Se admiten faltantes numéricos y categorías nuevas. Las columnas extra se ignoran y las columnas requeridas ausentes se rechazan.
- La pérdida esperada es PD × 0,5170492225611264 del monto original durante toda la vida del préstamo; no es una tasa anual ni una validación económica de precios.
- Cargar un archivo joblib puede ejecutar código. Usar solo el artefacto confiable del proyecto, nunca archivos de origen desconocido.

## Versiones y comandos

El JSON del artefacto registra Python **3.13.18**, scikit-learn **1.9.1**, LightGBM **4.7.0** y pandas **2.2.3**. La prueba de carga e inferencia pasó con Python **3.12.14** y las mismas versiones de esas tres librerías, además de NumPy 2.3.5, joblib 1.5.3, SciPy 1.17.0, threadpoolctl 3.6.0 y narwhals 2.28.0.

`requirements.txt` mantiene los mínimos generales del proyecto. `requirements-inference.txt` fija las versiones verificadas para la inferencia y las pruebas; no es un lock completo con hashes y no certifica la reproducción del entrenamiento. No se garantiza compatibilidad de la serialización con otras versiones. La diferencia de Python está documentada, no ocultada.

Desde la raíz del repositorio, en un entorno Python 3.12 aislado:

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
python -m pip install -r requirements-inference.txt
python -m unittest discover -s tests -v
```

Para ejecutar solo la prueba de persistencia en un proceso nuevo:

```bash
python -m unittest discover -s tests -p test_persistencia.py -v
```

La prueba carga el artefacto real y bloquea `fit` y `fit_transform` en Pipeline, ColumnTransformer, OneHotEncoder y LGBMClassifier. Usa cinco filas sintéticas, comprueba probabilidades finitas en [0, 1], suma por fila igual a 1, repetibilidad, esquema, bandas, columnas extra, orden de columnas y ausencia de una requerida. No descarga el dataset, no entrena y no evalúa calibración ni reproduce el AUC de test.

## Pérdidas observadas y regresiones

`tabla_por_banda` acepta pérdidas faltantes en préstamos pagados y las convierte en cero. **Un default sin pérdida informada genera `ValueError`**, para evitar subestimar silenciosamente la pérdida o promediar solo los defaults con información. También rechaza pérdidas infinitas o fuera de [0, 1], pérdidas positivas en pagados, objetivos no binarios, PD inválidas, bandas faltantes, longitudes incompatibles e índices de Series en distinto orden. Las entradas se interpretan en el mismo orden y no se realinean automáticamente.

Las pruebas verifican que los resultados válidos se conservan, que un default puede tener pérdida conocida igual a cero y que la función no modifica los datos recibidos. Las categorías sin préstamos se conservan con cantidad cero y medias faltantes.

Esta es una corrección preventiva. E02 informa 70.014 defaults y 70.014 pérdidas en train; no se verificó la completitud de las pérdidas de test. No hay evidencia de que el cambio altere las métricas publicadas. Se preservaron el modelo, sus metadatos, las celdas de código y todas las salidas numéricas del notebook; no se reejecutó el entrenamiento ni el test completo.

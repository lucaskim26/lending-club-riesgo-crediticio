"""Smoke test del artefacto confiable del repo: carga e inferencia, nunca fit.

Ejecutar en un proceso nuevo: python -m unittest discover -s tests -p test_persistencia.py -v
No descarga datos ni reentrena; no reproduce AUC ni valida calibración.
"""
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.evaluacion import asignar_banda
from src.modelado import cargar_modelo, predecir_pd


class PersistenciaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prohibiciones = ExitStack()
        cls.addClassCleanup(cls.prohibiciones.close)
        cls.llamadas_fit = []
        for clase in (Pipeline, ColumnTransformer, OneHotEncoder, LGBMClassifier):
            for metodo in ('fit', 'fit_transform'):
                if hasattr(clase, metodo):
                    mock = cls.prohibiciones.enter_context(patch.object(
                        clase, metodo, side_effect=AssertionError('El smoke test no permite entrenar.')))
                    cls.llamadas_fit.append(mock)
        cls.modelo, cls.meta = cargar_modelo()
        cls.numericas = cls.meta['variables']['numericas']
        cls.binarias = cls.meta['variables']['binarias']
        cls.categoricas = cls.meta['variables']['categoricas']
        cls.columnas = cls.numericas + cls.binarias + cls.categoricas
        base = {col: 0.0 for col in cls.numericas + cls.binarias}
        base.update(loan_amnt=10000, fico_promedio=702, log_annual_inc=float(np.log(60000)),
                    dti=15, open_acc=10, total_acc=20, revol_bal=5000, revol_util=35,
                    antiguedad_crediticia_anios=12, emp_length_anios=5,
                    prestamo_sobre_ingreso=10000/60000, revol_bal_sobre_ingreso=5000/60000,
                    home_ownership='RENT', purpose='debt_consolidation', addr_state='CA')
        filas = [base.copy() for _ in range(5)]
        filas[1].update(loan_amnt=25000, fico_promedio=662, dti=35,
                        log_annual_inc=float(np.log(30000)), prestamo_sobre_ingreso=25000/30000,
                        revol_util=90, home_ownership='MORTGAGE', purpose='credit_card', addr_state='NY')
        filas[2].update({col: np.nan for col in cls.numericas})
        filas[3].update({col: 'CATEGORIA_SINTETICA_NO_VISTA' for col in cls.categoricas})
        filas[4].update(loan_amnt=5000, fico_promedio=782, dti=5,
                        log_annual_inc=float(np.log(100000)), prestamo_sobre_ingreso=0.05,
                        home_ownership='OWN', purpose='home_improvement', addr_state='TX')
        cls.entrada = pd.DataFrame(filas, columns=cls.columnas)

    def tearDown(self):
        for mock in self.llamadas_fit:
            mock.assert_not_called()

    def test_pipeline_y_esquema_del_artefacto(self):
        self.assertIsInstance(self.modelo, Pipeline)
        self.assertIsInstance(self.modelo.named_steps['modelo'], LGBMClassifier)
        self.assertEqual(len(self.columnas), 54)
        prepro = self.modelo.named_steps['preprocesamiento']
        seleccionadas = [col for nombre, _, cols in prepro.transformers_
                        if nombre != 'remainder' for col in cols]
        self.assertEqual(seleccionadas, self.columnas)
        self.assertEqual(len(prepro.get_feature_names_out()), 102)
        self.assertFalse(self.meta['usa_desempleo'])

    def test_probabilidades_finitas_incluyendo_faltantes_y_categorias_nuevas(self):
        proba = self.modelo.predict_proba(self.entrada)
        self.assertEqual(proba.shape, (5, 2))
        self.assertTrue(np.isfinite(proba).all())
        self.assertTrue(((proba >= 0) & (proba <= 1)).all())
        np.testing.assert_allclose(proba.sum(axis=1), 1)
        np.testing.assert_array_equal(predecir_pd(self.modelo, self.entrada), proba[:, 1])

    def test_repeticion_y_columnas_extra_no_cambian_predicciones(self):
        pd_original = predecir_pd(self.modelo, self.entrada)
        np.testing.assert_array_equal(pd_original, predecir_pd(self.modelo, self.entrada))
        np.testing.assert_array_equal(
            pd_original, predecir_pd(self.modelo, self.entrada.assign(columna_extra='ignorada')))
        np.testing.assert_array_equal(
            pd_original, predecir_pd(self.modelo, self.entrada[self.columnas[::-1]]))

    def test_falta_de_columna_requerida_falla(self):
        with self.assertRaisesRegex(ValueError, 'fico_promedio'):
            predecir_pd(self.modelo, self.entrada.drop(columns='fico_promedio'))

    def test_bandas_y_perdida_esperada(self):
        cortes = np.asarray(self.meta['bandas']['cortes_pd'])
        nombres = self.meta['bandas']['nombres']
        self.assertTrue(np.isfinite(cortes).all())
        self.assertTrue((np.diff(cortes) > 0).all())
        self.assertEqual(len(cortes) + 1, len(nombres))
        self.assertTrue(((cortes > 0) & (cortes < 1)).all())
        severidad = self.meta['severidad_sobre_monto']
        self.assertTrue(np.isfinite(severidad) and 0 <= severidad <= 1)
        probabilidades = predecir_pd(self.modelo, self.entrada)
        bandas = asignar_banda(probabilidades, cortes, nombres)
        self.assertFalse(pd.isna(bandas).any())
        self.assertTrue(np.isfinite(probabilidades * severidad).all())
        self.assertEqual(list(asignar_banda(cortes, cortes, nombres)), nombres[1:])


if __name__ == '__main__':
    unittest.main()

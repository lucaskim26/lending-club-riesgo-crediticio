"""Regresiones de la tabla por banda, sin datos de entrenamiento."""
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.evaluacion import tabla_por_banda


class TablaPorBandaTests(unittest.TestCase):
    def setUp(self):
        self.bandas = pd.Categorical(['1', '1', '2'], categories=['1', '2', '3'], ordered=True)
        self.y = [0, 1, 1]
        self.p = [0.1, 0.3, 0.8]
        self.perdida = np.array([np.nan, 0.6, 0.8])

    def tabla(self, **cambios):
        args = dict(banda=self.bandas, y=self.y, pd_estimada=self.p,
                    perdida_sobre_monto=self.perdida, severidad=0.5)
        args.update(cambios)
        return tabla_por_banda(**args)

    def test_pagados_faltantes_son_cero_y_conserva_resultados_validos(self):
        tabla = self.tabla()
        self.assertEqual(tabla.loc['1', 'prestamos'], 2)
        self.assertAlmostEqual(tabla.loc['1', 'perdida_observada'], 0.3)
        self.assertAlmostEqual(tabla.loc['2', 'perdida_observada'], 0.8)
        self.assertAlmostEqual(tabla.loc['1', 'perdida_esperada'], 0.1)
        self.assertAlmostEqual(tabla['% de prestamos'].sum(), 1)
        self.assertEqual(tabla.loc['3', 'prestamos'], 0)
        self.assertTrue(np.isnan(tabla.loc['3', 'perdida_observada']))
        self.assertTrue(np.isnan(self.perdida[0]))  # No modifica la entrada.
        assert_frame_equal(tabla, self.tabla(perdida_sobre_monto=[0, 0.6, 0.8]))

    def test_default_con_perdida_faltante_falla(self):
        for faltante in [np.nan, None, pd.NA]:
            with self.subTest(faltante=faltante), self.assertRaisesRegex(ValueError, 'defaults.*faltante'):
                self.tabla(perdida_sobre_monto=pd.Series([0, faltante, 0.8], dtype='Float64'))

    def test_perdida_invalida_falla(self):
        for perdida in [np.inf, -np.inf, -0.1, 1.1]:
            for fila in [0, 1]:
                valores = self.perdida.copy()
                valores[fila] = perdida
                with self.subTest(perdida=perdida, fila=fila), self.assertRaisesRegex(ValueError, 'pérdida informada'):
                    self.tabla(perdida_sobre_monto=valores)

    def test_pagado_con_perdida_positiva_falla(self):
        with self.assertRaisesRegex(ValueError, 'pagado debe ser 0'):
            self.tabla(perdida_sobre_monto=[0.2, 0.6, 0.8])

    def test_default_con_perdida_cero_es_valido(self):
        tabla = self.tabla(perdida_sobre_monto=[np.nan, 0, 1])
        self.assertEqual(tabla.loc['1', 'perdida_observada'], 0)
        self.assertEqual(tabla.loc['2', 'perdida_observada'], 1)

    def test_perdida_y_severidad_son_opcionales(self):
        tabla = self.tabla(perdida_sobre_monto=None, severidad=None)
        self.assertNotIn('perdida_observada', tabla)
        self.assertNotIn('perdida_esperada', tabla)

    def test_longitudes_y_dimensiones_invalidas_fallan(self):
        for campo in ['banda', 'y', 'pd_estimada', 'perdida_sobre_monto']:
            for valor in [[0], [[0], [1], [1]], 0.5]:
                with self.subTest(campo=campo, valor=valor), self.assertRaises(ValueError):
                    self.tabla(**{campo: valor})
        with self.assertRaises(ValueError):
            tabla_por_banda([], [], [])

    def test_objetivo_invalido_falla(self):
        for valor in [np.nan, np.inf, -1, 0.5, 2]:
            with self.subTest(valor=valor), self.assertRaisesRegex(ValueError, 'y debe contener'):
                self.tabla(y=[0, valor, 1])

    def test_probabilidades_invalidas_fallan(self):
        for valor in [np.nan, np.inf, -0.1, 1.1]:
            with self.subTest(valor=valor), self.assertRaisesRegex(ValueError, 'PD debe ser finita'):
                self.tabla(pd_estimada=[valor, 0.3, 0.8])

    def test_banda_faltante_falla_en_vez_de_omitir_prestamo(self):
        with self.assertRaisesRegex(ValueError, 'banda, sin faltantes'):
            self.tabla(banda=['1', None, '2'])

    def test_severidad_invalida_falla(self):
        for valor in [np.nan, np.inf, -0.1, 1.1, [0.5]]:
            with self.subTest(valor=valor), self.assertRaisesRegex(ValueError, 'severidad'):
                self.tabla(severidad=valor)

    def test_series_con_indice_desordenado_falla(self):
        with self.assertRaisesRegex(ValueError, 'índices'):
            self.tabla(y=pd.Series(self.y, index=[20, 10, 30]),
                       perdida_sobre_monto=pd.Series(self.perdida, index=[10, 20, 30]))

    def test_series_alineadas_con_indice_no_consecutivo(self):
        indice = [20, 10, 30]
        tabla = self.tabla(banda=pd.Series(self.bandas, index=indice),
                           y=pd.Series(self.y, index=indice),
                           pd_estimada=pd.Series(self.p, index=indice),
                           perdida_sobre_monto=pd.Series(self.perdida, index=indice))
        assert_frame_equal(tabla, self.tabla())

    def test_regresion_vs_calculo_anterior_con_perdidas_completas(self):
        # El cálculo anterior era correcto si todos los defaults tenían pérdida.
        datos = pd.DataFrame({'banda': self.bandas, 'pd': self.p, 'default': self.y,
                              'perdida': np.nan_to_num(self.perdida, nan=0)})
        esperada = datos.groupby('banda', observed=False).agg(
            prestamos=('pd', 'size'), pd_media=('pd', 'mean'),
            default_observado=('default', 'mean'), perdida_observada=('perdida', 'mean'))
        esperada.insert(1, '% de prestamos', esperada['prestamos'] / esperada['prestamos'].sum())
        esperada['perdida_esperada'] = esperada['pd_media'] * 0.5
        assert_frame_equal(self.tabla(), esperada)


if __name__ == '__main__':
    unittest.main()

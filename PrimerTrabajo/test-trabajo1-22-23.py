import unittest
import importlib
import numpy as np

file_name = "trabajo1-22-23"

trabajo = importlib.import_module(file_name)
from datos_trabajo_aa import carga_datos as datos

class TestExercise1(unittest.TestCase):

    def test_particion_entr_prueba(self):
        print("\n[ #1 ] Running test_particion_entr_prueba...")
        for i in range(1, 8):
            split = i/10
            # Crédito
            X_credito, y_credito = datos.X_credito, datos.y_credito
            Xe_credito,Xp_credito,ye_credito,yp_credito = trabajo.particion_entr_prueba(X_credito,y_credito,test=split)
            self.assertTrue(abs(Xe_credito.shape[0] - X_credito.shape[0]*(1-split)) < 1)
            self.assertTrue(abs(Xp_credito.shape[0] - X_credito.shape[0]*split) < 1)
            self.assertTrue(abs(ye_credito.shape[0] - y_credito.shape[0]*(1-split)) < 1)
            self.assertTrue(abs(yp_credito.shape[0] - y_credito.shape[0]*split) < 1)
            # Test that all elements in X_credito are in Xe_credito or Xp_credito
            self.assertTrue(np.all(np.isin(X_credito, np.concatenate((Xe_credito, Xp_credito)))))
            self.assertTrue(np.all(np.isin(y_credito, np.concatenate((ye_credito, yp_credito)))))

            # Votos
            X_votos, y_votos = datos.X_votos, datos.y_votos
            Xe_votos,Xp_votos,ye_votos,yp_votos = trabajo.particion_entr_prueba(X_votos,y_votos,test=split)
            self.assertTrue(abs(Xe_votos.shape[0] - X_votos.shape[0]*(1-split)) < 1)
            self.assertTrue(abs(Xp_votos.shape[0] - X_votos.shape[0]*split) < 1)
            self.assertTrue(abs(ye_votos.shape[0] - y_votos.shape[0]*(1-split)) < 1)
            self.assertTrue(abs(yp_votos.shape[0] - y_votos.shape[0]*split) < 1)
            # Test that all elements in X_votos are in Xe_votos or Xp_votos
            self.assertTrue(np.all(np.isin(X_votos, np.concatenate((Xe_votos, Xp_votos)))))
            self.assertTrue(np.all(np.isin(y_votos, np.concatenate((ye_votos, yp_votos)))))

            # Cancer
            X_cancer, y_cancer = datos.X_cancer, datos.y_cancer
            Xe_cancer,Xp_cancer,ye_cancer,yp_cancer = trabajo.particion_entr_prueba(X_cancer,y_cancer,test=split)
            self.assertTrue(abs(Xe_cancer.shape[0] - X_cancer.shape[0]*(1-split)) < 1)
            self.assertTrue(abs(Xp_cancer.shape[0] - X_cancer.shape[0]*split) < 1)
            self.assertTrue(abs(ye_cancer.shape[0] - y_cancer.shape[0]*(1-split)) < 1)
            self.assertTrue(abs(yp_cancer.shape[0] - y_cancer.shape[0]*split) < 1)
            # Test that all elements in X_cancer are in Xe_cancer or Xp_cancer
            self.assertTrue(np.all(np.isin(X_cancer, np.concatenate((Xe_cancer, Xp_cancer)))))
            self.assertTrue(np.all(np.isin(y_cancer, np.concatenate((ye_cancer, yp_cancer)))))
        # Print a checkmark in green color
        print(u'\u2714', "test_particion_entr_prueba passed\n\n")


class TestExercise2(unittest.TestCase):
    def test_RegresionLogisticaMiniBatch(self):
        print("\n[ #2 ] Running test_RegresionLogisticaMiniBatch...")
        # Votos tests
        Xe_votos, Xp_votos, ye_votos, yp_votos = trabajo.particion_entr_prueba(datos.X_votos, datos.y_votos)

        RLMB_votos = trabajo.RegresionLogisticaMiniBatch(normalizacion = True, rate_decay = True, batch_tam = 12)
        self.assertRaises(Exception, RLMB_votos.clasifica, Xp_votos)
        RLMB_votos.entrena(Xe_votos, ye_votos)
        classification = RLMB_votos.clasifica(Xp_votos)
        self.assertEqual(classification.shape, yp_votos.shape)
        performance = trabajo.rendimiento(RLMB_votos, Xp_votos, yp_votos)
        print("Votos performance: ", performance)
        self.assertTrue(performance > 0.75)

        # Cancer tests
        Xe_cancer, Xp_cancer, ye_cancer, yp_cancer = trabajo.particion_entr_prueba(datos.X_cancer, datos.y_cancer)

        RLMB_cancer = trabajo.RegresionLogisticaMiniBatch(normalizacion=True, rate_decay=True)
        self.assertRaises(Exception, RLMB_cancer.clasifica, Xp_cancer)
        RLMB_cancer.entrena(Xe_cancer, ye_cancer)
        classification = RLMB_cancer.clasifica(Xp_cancer)
        self.assertEqual(classification.shape, yp_cancer.shape)
        performance = trabajo.rendimiento(RLMB_cancer, Xp_cancer, yp_cancer)
        print("Cancer performance: ", performance)
        self.assertTrue(performance > 0.75)

        # Crédito tests
        Xe_credito, Xp_credito, ye_credito, yp_credito = trabajo.particion_entr_prueba(datos.X_credito, datos.y_credito)

        RLMB_credito = trabajo.RegresionLogisticaMiniBatch()
        self.assertRaises(Exception, RLMB_credito.clasifica, Xp_credito)
        self.assertRaises(ValueError, RLMB_credito.entrena, Xe_credito, ye_credito)

        print(u'\n\n\u2714', "test_RegresionLogisticaMiniBatch passed\n\n")

class TestExercise3(unittest.TestCase):
    def test_rendimiento_validacion_cruzada(self):
        print("\n[ #3 ] Running test_rendimiento_validacion_cruzada...")
        # Votos tests
        Xe_votos, _, ye_votos, _ = trabajo.particion_entr_prueba(datos.X_votos, datos.y_votos)

        params = {'normalizacion': True, 'rate_decay': True, 'batch_tam': 16}
        performance = trabajo.rendimiento_validacion_cruzada(trabajo.RegresionLogisticaMiniBatch, params, Xe_votos, ye_votos, 3)
        print("Votos performance: ", performance)
        self.assertTrue(performance > 0.75)

        # Cancer tests
        Xe_cancer, _, ye_cancer, _ = trabajo.particion_entr_prueba(datos.X_cancer, datos.y_cancer)

        params = {'normalizacion': True, 'rate_decay': True, 'batch_tam': 16}
        performance = trabajo.rendimiento_validacion_cruzada(trabajo.RegresionLogisticaMiniBatch, params, Xe_cancer, ye_cancer, 2)
        print("Cancer performance: ", performance)
        self.assertTrue(performance > 0.75)

        print(u'\n\n\u2714', "test_rendimiento_validacion_cruzada passed\n\n")


if __name__ == '__main__':
    unittest.main()
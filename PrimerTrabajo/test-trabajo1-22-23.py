import unittest
import importlib
import numpy as np

file_name = "trabajo1-22-23"

class TestExercise1(unittest.TestCase):
    trabajo = importlib.import_module(file_name)
    from datos_trabajo_aa import carga_datos as datos

    def test_particion_entr_prueba(self):
        for i in range(1, 8):
            split = i/10
            # Crédito
            X_credito, y_credito = self.datos.X_credito, self.datos.y_credito
            Xe_credito,Xp_credito,ye_credito,yp_credito = self.trabajo.particion_entr_prueba(X_credito,y_credito,test=split)
            self.assertTrue(abs(Xe_credito.shape[0] - X_credito.shape[0]*(1-split)) < 1)
            self.assertTrue(abs(Xp_credito.shape[0] - X_credito.shape[0]*split) < 1)
            self.assertTrue(abs(ye_credito.shape[0] - y_credito.shape[0]*(1-split)) < 1)
            self.assertTrue(abs(yp_credito.shape[0] - y_credito.shape[0]*split) < 1)
            # Test that all elements in X_credito are in Xe_credito or Xp_credito
            self.assertTrue(np.all(np.isin(X_credito, np.concatenate((Xe_credito, Xp_credito)))))

            # Votos
            X_votos, y_votos = self.datos.X_votos, self.datos.y_votos
            Xe_votos,Xp_votos,ye_votos,yp_votos = self.trabajo.particion_entr_prueba(X_votos,y_votos,test=split)
            self.assertTrue(abs(Xe_votos.shape[0] - X_votos.shape[0]*(1-split)) < 1)
            self.assertTrue(abs(Xp_votos.shape[0] - X_votos.shape[0]*split) < 1)
            self.assertTrue(abs(ye_votos.shape[0] - y_votos.shape[0]*(1-split)) < 1)
            self.assertTrue(abs(yp_votos.shape[0] - y_votos.shape[0]*split) < 1)
            # Test that all elements in X_votos are in Xe_votos or Xp_votos
            self.assertTrue(np.all(np.isin(X_votos, np.concatenate((Xe_votos, Xp_votos)))))

            # Cancer
            X_cancer, y_cancer = self.datos.X_cancer, self.datos.y_cancer
            Xe_cancer,Xp_cancer,ye_cancer,yp_cancer = self.trabajo.particion_entr_prueba(X_cancer,y_cancer,test=split)
            self.assertTrue(abs(Xe_cancer.shape[0] - X_cancer.shape[0]*(1-split)) < 1)
            self.assertTrue(abs(Xp_cancer.shape[0] - X_cancer.shape[0]*split) < 1)
            self.assertTrue(abs(ye_cancer.shape[0] - y_cancer.shape[0]*(1-split)) < 1)
            self.assertTrue(abs(yp_cancer.shape[0] - y_cancer.shape[0]*split) < 1)
            # Test that all elements in X_cancer are in Xe_cancer or Xp_cancer
            self.assertTrue(np.all(np.isin(X_cancer, np.concatenate((Xe_cancer, Xp_cancer)))))



if __name__ == '__main__':
    unittest.main()

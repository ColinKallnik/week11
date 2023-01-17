from sklearn.datasets import load_wine
from wine_model import train_model
import numpy as np

def test_wine():
   """Predictions result in 0, 1 or 2"""
   X, y = load_wine(return_X_y=True)
   m = train_model(X, y)
   ypred = m.predict(X)
   # check that all predictions are 0, 1 or 2
   assert np.all((ypred == 0) | (ypred == 1) | (ypred == 2))
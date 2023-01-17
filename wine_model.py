from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
import pandas as pd

d = load_wine()
print(d['DESCR'])
X = pd.DataFrame(d['data'], columns=d['feature_names'])
y = d['target']  # cultivator

def train_model(X, y):
   model = LogisticRegression
   model.fit(X, y)
   return model
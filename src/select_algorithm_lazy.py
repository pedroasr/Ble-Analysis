import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# Carga de datos.
trainingSet = pd.read_csv("../results/training/filled-training-set.csv", sep=";")

# División de datos.
X = trainingSet.loc[:, (trainingSet.columns != "Timestamp") & (trainingSet.columns != "Ocupacion")]
y = trainingSet.loc[:, trainingSet.columns == "Ocupacion"]

# División de datos en entrenamiento y test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=288)

# Entrenamiento de modelos.
reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)

models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print(models)

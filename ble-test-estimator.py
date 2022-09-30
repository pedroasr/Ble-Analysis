import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

trainingSet = pd.read_csv("docs/filled-training-set.csv", sep=";")

x = trainingSet.loc[:, (trainingSet.columns != "Timestamp") & (trainingSet.columns != "Person Count")]
y = trainingSet["Person Count"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=288)

reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print(models)

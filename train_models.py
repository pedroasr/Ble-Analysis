import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
import lightgbm as ltb
import warnings

warnings.filterwarnings("ignore")

pd.options.mode.chained_assignment = None

trainingSet = pd.read_csv("../docs/filled-training-set.csv", sep=";")

X = trainingSet.loc[:, (trainingSet.columns != "Timestamp") & (trainingSet.columns != "Ocupacion")]
y = trainingSet.loc[:, trainingSet.columns == "Ocupacion"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=288)

# Primera opción de estimador -- HistGradientBoostingRegressor
est = HistGradientBoostingRegressor().fit(X_train, y_train)
print(f"R Square of HistGradientBoostingRegressor: {est.score(X_test, y_test):0.4f}")
joblib.dump(est, '../models/HistGradientBoostingRegressor.pkl')

# Segunda opción de estimador -- LGBMRegressor
lgbm = ltb.LGBMRegressor().fit(X_train, y_train)
print(f"R Square of LGBMRegressor: {lgbm.score(X_test, y_test):0.4f}")
joblib.dump(lgbm, '../models/LGBMRegressor.pkl')

# Tercera opción de estimador -- RandomForestRegressor
# rfr = RandomForestRegressor().fit(X_train, y_train)
# print(f"R Square of RandomForestRegressor: {rfr.score(X_test, y_test):0.4f}")
# joblib.dump(rfr, '../models/RandomForestRegressor.pkl')

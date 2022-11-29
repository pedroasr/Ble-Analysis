import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

trainingSet = pd.read_csv("../docs/filled-training-set.csv", sep=";")

X = trainingSet.loc[:, (trainingSet.columns != "Timestamp") & (trainingSet.columns != "Ocupacion")]
y = trainingSet.loc[:, trainingSet.columns == "Ocupacion"]

# Primera opción de estimador -- RandomForestRegressor
rfr = RandomForestRegressor().fit(X, y)
joblib.dump(rfr, '../models/RandomForestRegressor.pkl')

# Segunda opción de estimador -- ExtraTreesRegressor
etr = ExtraTreesRegressor().fit(X, y)
joblib.dump(etr, '../models/ExtraTreesRegressor.pkl')

# Tercera opción de estimador -- XGBRegressor
xgbm = xgb.XGBRegressor().fit(X, y)
joblib.dump(xgbm, '../models/XGBRegressor.pkl')

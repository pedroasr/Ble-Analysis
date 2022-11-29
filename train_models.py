import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import lightgbm as ltb
import warnings

warnings.filterwarnings("ignore")

trainingSet = pd.read_csv("../docs/filled-training-set.csv", sep=";")

X = trainingSet.loc[:, (trainingSet.columns != "Timestamp") & (trainingSet.columns != "Ocupacion")]
y = trainingSet.loc[:, trainingSet.columns == "Ocupacion"]

# Primera opción de estimador -- ExtraTreesRegressor
etr = ExtraTreesRegressor().fit(X, y)
joblib.dump(etr, '../models/ExtraTreesRegressor.pkl')

# Segunda opción de estimador -- RandomForestRegressor
rfr = RandomForestRegressor().fit(X, y)
joblib.dump(rfr, '../models/RandomForestRegressor.pkl')

# Tercera opción de estimador -- LGBMRegressor
lgbm = ltb.LGBMRegressor().fit(X, y)
joblib.dump(lgbm, '../models/LGBMRegressor.pkl')

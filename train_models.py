import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import xgboost as xgb
from scipy.stats import uniform
import warnings

warnings.filterwarnings("ignore")

trainingSet = pd.read_csv("../docs/filled-training-set.csv", sep=";")

X = trainingSet.loc[:, (trainingSet.columns != "Timestamp") & (trainingSet.columns != "Ocupacion")]
y = trainingSet.loc[:, trainingSet.columns == "Ocupacion"]

# Primera opción de estimador -- RandomForestRegressor
print("Comenzando entrenamiento de RandomForestRegressor")
rfr = RandomForestRegressor()
distributionRfr = dict(n_estimators=[int(x) for x in range(100, 1000, 100)])
clfRfr = RandomizedSearchCV(rfr, distributionRfr, random_state=0, cv=5)
searchRfr = clfRfr.fit(X, y)
bestParamsRfr = searchRfr.best_params_
bestScoreRfr = searchRfr.best_score_
print(f"Best param of n_estimators: {bestParamsRfr['n_estimators']:0.4f}")
print(f"R Square of RandomForestRegressor: {bestScoreRfr:0.4f}")
bestEstimatorRfr = RandomForestRegressor(n_estimators=bestParamsRfr["n_estimators"])
modelRfr = bestEstimatorRfr.fit(X, y)
joblib.dump(modelRfr, '../models/RandomForestRegressor.pkl')

# Segunda opción de estimador -- ExtraTreesRegressor
print("Comenzando entrenamiento de ExtraTreesRegressor")
etr = ExtraTreesRegressor()
distributionEtr = dict(n_estimators=[int(x) for x in range(100, 1000, 100)])
clfEtr = RandomizedSearchCV(etr, distributionEtr, random_state=0, cv=5)
searchEtr = clfEtr.fit(X, y)
bestParamsEtr = searchEtr.best_params_
bestScoreEtr = searchEtr.best_score_
print(f"Best param of n_estimators: {bestParamsEtr['n_estimators']:0.4f}")
print(f"R Square of ExtraTreesRegressor: {bestScoreEtr:0.4f}")
bestEstimatorEtr = ExtraTreesRegressor(n_estimators=bestParamsEtr["n_estimators"])
modelEtr = bestEstimatorEtr.fit(X, y)
joblib.dump(modelEtr, '../models/ExtraTreesRegressor.pkl')

# Tercera opción de estimador -- XGBRegressor
print("Comenzando entrenamiento de XGBRegressor")
xgbm = xgb.XGBRegressor()
distributionXgbm = dict(eta=uniform(), max_depth=[int(x) for x in range(3, 10)], subsample=uniform())
clfXgbm = RandomizedSearchCV(xgbm, distributionXgbm, random_state=0, cv=5)
searchXgbm = clfXgbm.fit(X, y)
bestParamsXgbm = searchXgbm.best_params_
bestScoreXgbm = searchXgbm.best_score_
print(
    f"Best param of eta: {bestParamsEtr['eta']:0.4f}, max_depth: {bestParamsEtr['max_depth']:0.4f}, subsample: {bestParamsEtr['subsample']:0.4f}")
print(f"R Square of XGBRegressor: {bestScoreXgbm:0.4f}")
bestEstimatorXgbm = xgb.XGBRegressor(eta=bestParamsXgbm["eta"], max_depth=bestParamsXgbm["max_depth"],
                                     subsample=bestParamsXgbm["subsample"])
modelXgbm = bestEstimatorXgbm.fit(X, y)
joblib.dump(modelXgbm, '../models/XGBRegressor.pkl')

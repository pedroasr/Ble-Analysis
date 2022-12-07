import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import xgboost as xgb
from scipy.stats import uniform
import warnings

warnings.filterwarnings("ignore")


def trainModels(path, folder):
    """Función que carga los datos de aprendizaje, busca los mejores hiperparámetros para cada modelo y los guarda en la
    carpeta destino."""

    # Carga de datos.
    trainingSet = pd.read_csv(path, sep=";")

    # Separación de datos.
    X = trainingSet.loc[:, (trainingSet.columns != "Timestamp") & (trainingSet.columns != "Ocupacion")]
    y = trainingSet.loc[:, trainingSet.columns == "Ocupacion"]

    # Primera opción de estimador -- RandomForestRegressor
    print("Comenzando entrenamiento de RandomForestRegressor")

    # Se busca la mejor combinación de hiperparámetros.
    rfr = RandomForestRegressor()
    distributionRfr = dict(n_estimators=[int(x) for x in range(100, 1000, 100)])
    clfRfr = RandomizedSearchCV(rfr, distributionRfr, random_state=0, cv=5)
    searchRfr = clfRfr.fit(X, y)

    # Se recupera la mejor combinación de hiperparámetros.
    bestParamsRfr = searchRfr.best_params_
    bestScoreRfr = searchRfr.best_score_
    print(f"Best param of n_estimators: {bestParamsRfr['n_estimators']}")
    print(f"R Square of RandomForestRegressor: {bestScoreRfr:0.4f}")
    print("")
    bestEstimatorRfr = RandomForestRegressor(n_estimators=bestParamsRfr["n_estimators"])

    # Se entrena el modelo con la mejor combinación de hiperparámetros y todos los datos disponibles.
    modelRfr = bestEstimatorRfr.fit(X, y)
    joblib.dump(modelRfr, f"{folder}/RandomForestRegressor.pkl")

    # Segunda opción de estimador -- ExtraTreesRegressor
    print("Comenzando entrenamiento de ExtraTreesRegressor")

    # Se busca la mejor combinación de hiperparámetros.
    etr = ExtraTreesRegressor()
    distributionEtr = dict(n_estimators=[int(x) for x in range(100, 1000, 100)])
    clfEtr = RandomizedSearchCV(etr, distributionEtr, random_state=0, cv=5)
    searchEtr = clfEtr.fit(X, y)

    # Se recupera la mejor combinación de hiperparámetros.
    bestParamsEtr = searchEtr.best_params_
    bestScoreEtr = searchEtr.best_score_
    print(f"Best param of n_estimators: {bestParamsEtr['n_estimators']}")
    print(f"R Square of ExtraTreesRegressor: {bestScoreEtr:0.4f}")
    print("")

    # Se entrena el modelo con la mejor combinación de hiperparámetros y todos los datos disponibles.
    bestEstimatorEtr = ExtraTreesRegressor(n_estimators=bestParamsEtr["n_estimators"])
    modelEtr = bestEstimatorEtr.fit(X, y)
    joblib.dump(modelEtr, f"{folder}/ExtraTreesRegressor.pkl")

    # Tercera opción de estimador -- XGBRegressor
    print("Comenzando entrenamiento de XGBRegressor")

    # Se busca la mejor combinación de hiperparámetros.
    xgbm = xgb.XGBRegressor()
    distributionXgbm = dict(eta=uniform(), max_depth=[int(x) for x in range(3, 10)], subsample=uniform())
    clfXgbm = RandomizedSearchCV(xgbm, distributionXgbm, random_state=0, cv=5)
    searchXgbm = clfXgbm.fit(X, y)

    # Se recupera la mejor combinación de hiperparámetros.
    bestParamsXgbm = searchXgbm.best_params_
    bestScoreXgbm = searchXgbm.best_score_
    print(
        f"Best param of eta: {bestParamsXgbm['eta']:0.2f}, max_depth: {bestParamsXgbm['max_depth']:0.2f}, subsample: "
        f"{bestParamsXgbm['subsample']:0.2f}")
    print(f"R Square of XGBRegressor: {bestScoreXgbm:0.4f}")

    # Se entrena el modelo con la mejor combinación de hiperparámetros y todos los datos disponibles.
    bestEstimatorXgbm = xgb.XGBRegressor(eta=bestParamsXgbm["eta"], max_depth=bestParamsXgbm["max_depth"],
                                         subsample=bestParamsXgbm["subsample"])
    modelXgbm = bestEstimatorXgbm.fit(X, y)
    joblib.dump(modelXgbm, f"{folder}/XGBRegressor.pkl")

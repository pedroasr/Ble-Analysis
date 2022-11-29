import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import lightgbm as ltb


data = pd.read_csv("../docs/filled-test-set.csv", sep=";")

data["Fecha"] = data["Timestamp"].apply(lambda x: x.split(" ")[0])
fechas = data["Fecha"].unique()

for i in range(len(fechas)):

    trainingData = data.loc[data["Fecha"] != fechas[i]]
    testData = data.loc[data["Fecha"] == fechas[i]]
    testData["Timestamp"] = pd.to_datetime(testData["Timestamp"])

    X_train = trainingData.loc[:, (trainingData.columns != "Timestamp") & (trainingData.columns != "Ocupacion") & (trainingData.columns != "Fecha")]
    y_train = np.ravel(trainingData.loc[:, trainingData.columns == "Ocupacion"])

    X_test = testData.loc[:, (testData.columns != "Timestamp") & (testData.columns != "Ocupacion") & (testData.columns != "Fecha")]
    y_test = np.ravel(testData.loc[:, testData.columns == "Ocupacion"])

    # Primera opción de estimador -- HistGradientBoostingRegressor
    est = HistGradientBoostingRegressor().fit(X_train, y_train)
    print(f"R Square of HistGradientBoostingRegressor: {est.score(X_test, y_test):0.4f}")
    # joblib.dump(est, '../models/HistGradientBoostingRegressor.pkl')

    # Segunda opción de estimador -- LGBMRegressor
    lgbm = ltb.LGBMRegressor().fit(X_train, y_train)
    print(f"R Square of LGBMRegressor: {lgbm.score(X_test, y_test):0.4f}")
    # joblib.dump(lgbm, '../models/LGBMRegressor.pkl')

    # Tercera opción de estimador -- RandomForestRegressor
    # rfr = RandomForestRegressor().fit(X_train, y_train)
    # print(f"R Square of RandomForestRegressor: {rfr.score(X_test, y_test):0.4f}")
    # joblib.dump(rfr, '../models/RandomForestRegressor.pkl')

    name = fechas[i].replace("/", "-")

    plt.figure(figsize=(10, 6))
    plt.plot(testData["Timestamp"], y_test, label="Real")
    plt.plot(testData["Timestamp"], est.predict(X_test), label="HistGradientBoostingRegressor")
    plt.plot(testData["Timestamp"], lgbm.predict(X_test), label="LGBMRegressor")
    plt.xlabel("Timestamp")
    plt.ylabel("Ocupacion")
    plt.title(name)
    plt.legend()
    plt.savefig("../figuresPredict/" + name + '.jpg')
    plt.clf()
    plt.close()

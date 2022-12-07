import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# Carga de datos.
trainingSet = pd.read_csv("../docs/filled-training-set.csv", sep=";")
trainingSet["Timestamp"] = pd.to_datetime(trainingSet["Timestamp"])
dates = trainingSet["Timestamp"].dt.strftime('%Y-%m-%d %H:%M:%S')

# División de datos.
X = trainingSet.loc[:, (trainingSet.columns != "Timestamp") & (trainingSet.columns != "Ocupacion")]
y = trainingSet.loc[:, trainingSet.columns == "Ocupacion"]

# Creación de modelos con los hiperparámetros óptimos.
est = ExtraTreesRegressor(n_estimators=500)
rdf = RandomForestRegressor(n_estimators=600)
xgbm = xgb.XGBRegressor(eta=0.14, max_depth=3, subsample=0.98)

# Usando Cross Validation, devuelve los valores predichos.
estPredict = cross_val_predict(est, X, y, cv=5).astype(int)
rdfPredict = cross_val_predict(rdf, X, y, cv=5).astype(int)
xgbmPredict = cross_val_predict(xgbm, X, y, cv=5).astype(int)

# Se guardan los datos en un DataFrame.
data = np.array(np.transpose([dates, y["Ocupacion"], estPredict, rdfPredict, xgbmPredict]))
predDataframe = pd.DataFrame(data=data, columns=["Timestamp", "Ocupacion", "ExtraTreesRegressor", "RandomForestRegressor", "XGBRegressor"])
predDataframe["Timestamp"] = pd.to_datetime(predDataframe["Timestamp"])

# Se grafican los datos para cada día.
for date in trainingSet["Timestamp"].dt.date.unique():
    group = predDataframe.loc[predDataframe["Timestamp"].dt.date == date]
    name = date.strftime("%Y-%m-%d")

    fig, ax = plt.subplots(figsize=(10, 6))
    date_form = DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_form)
    plt.plot(group["Timestamp"], group["Ocupacion"], label="Ocupacion", color="red")
    plt.plot(group["Timestamp"], group["ExtraTreesRegressor"], label="ExtraTreesRegressor", color="blue")
    plt.plot(group["Timestamp"], group["RandomForestRegressor"], label="RandomForestRegressor", color="green")
    plt.plot(group["Timestamp"], group["XGBRegressor"], label="XGBRegressor", color="yellow")
    plt.xlabel("Hora")
    plt.ylabel("Ocupacion")
    plt.legend()
    plt.grid()
    plt.title(name)
    plt.savefig("../figuresCrossValPred/" + name + ".jpg")
    plt.clf()
    plt.close()

import pandas as pd
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

path1 = Path("../figures/CrossValPred/predictions")
path2 = Path("../figures/CrossValPred/error")

if not path1.exists():
    path1.mkdir(parents=True)
if not path2.exists():
    path2.mkdir(parents=True)

# Carga de datos.
trainingSet = pd.read_csv("../results/training/filled-training-set.csv", sep=";")
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

# Se calcula el error absoluto.
estError = np.absolute(estPredict - y["Ocupacion"]).astype(int)
rdfError = np.absolute(rdfPredict - y["Ocupacion"]).astype(int)
xgbmError = np.absolute(xgbmPredict - y["Ocupacion"]).astype(int)

# Se guardan los datos en un DataFrame.
data = np.array(
    np.transpose([dates, y["Ocupacion"], estPredict, rdfPredict, xgbmPredict, estError, rdfError, xgbmError]))
predDataframe = pd.DataFrame(data=data,
                             columns=["Timestamp", "Ocupacion", "ExtraTreesRegressor", "RandomForestRegressor",
                                      "XGBRegressor", "ErrorExtraTreesRegressor", "ErrorRandomForestRegressor",
                                      "ErrorXGBRegressor"])

predDataframe["Timestamp"] = pd.to_datetime(predDataframe["Timestamp"])

# Se grafican los datos para cada día.
for date in trainingSet["Timestamp"].dt.date.unique():
    group = predDataframe.loc[predDataframe["Timestamp"].dt.date == date]
    name = date.strftime("%Y-%m-%d")

    print("Graficando " + name + "...")
    print("")
    print(f"El error máximo de ExtraTreesRegressor es: {group['ErrorExtraTreesRegressor'].max()}")
    print(f"El error máximo de RandomForestRegressor es: {group['ErrorRandomForestRegressor'].max()}")
    print(f"El error máximo de XGBRegressor es: {group['ErrorXGBRegressor'].max()}")
    print("")
    print(f"El percentil 75 de ExtraTreesRegressor es: {np.percentile(group['ErrorExtraTreesRegressor'], 75)}")
    print(f"El percentil 75 de RandomForestRegressor es: {np.percentile(group['ErrorRandomForestRegressor'], 75)}")
    print(f"El percentil 75 de XGBRegressor es: {np.percentile(group['ErrorXGBRegressor'], 75)}")
    print("")

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
    plt.savefig(Path(path1, name + ".jpg"))
    plt.clf()

    plt.plot(group["Timestamp"], group["Ocupacion"], label="Ocupacion", color="red")
    plt.plot(group["Timestamp"], group["ErrorExtraTreesRegressor"], label="ExtraTreesRegressor", color="blue")
    plt.plot(group["Timestamp"], group["ErrorRandomForestRegressor"], label="RandomForestRegressor", color="green")
    plt.plot(group["Timestamp"], group["ErrorXGBRegressor"], label="XGBRegressor", color="yellow")
    plt.xlabel("Hora")
    plt.ylabel("Error")
    plt.legend()
    plt.grid()
    plt.title(name)
    plt.savefig(Path(path2, name + ".jpg"))
    plt.clf()
    plt.close()

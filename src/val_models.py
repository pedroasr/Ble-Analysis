from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
import warnings

warnings.filterwarnings("ignore")


def valModels(pathTestSet, pathExtraTree, pathXGB, pathRandomForest, folder):
    """Función que carga los modelos entrenados y los aplica al conjunto de test."""

    folder = Path("../figures", folder)
    if not folder.exists():
        folder.mkdir(parents=True)

    pathPred = Path(folder, "prediction")
    pathError = Path(folder, "error")
    if not pathPred.exists():
        pathPred.mkdir(parents=True)
    if not pathError.exists():
        pathError.mkdir(parents=True)

    # Cargamos los datos.
    testSet = pd.read_csv(pathTestSet, sep=";")

    # Cargamos los modelos.
    est = joblib.load(pathExtraTree)
    xgbm = joblib.load(pathXGB)
    rfr = joblib.load(pathRandomForest)

    testSet["Timestamp"] = pd.to_datetime(testSet["Timestamp"])
    dates = testSet["Timestamp"].dt.date.unique()
    dataArray = []

    # Para cada fecha disponible, se genera un gráfico con los datos reales y los predichos además del error producido.
    for date in dates:
        group = testSet.loc[testSet["Timestamp"].dt.date == date]
        data = pd.DataFrame(group[["Timestamp", "Ocupacion"]], columns=["Timestamp", "Ocupacion"])

        name = date.strftime("%Y-%m-%d")

        X = group.loc[:, (group.columns != "Timestamp") & (group.columns != "Ocupacion")]

        # Se aplican los modelos.
        predicted_etr_y = est.predict(X)
        predicted_xgbm_y = xgbm.predict(X)
        predicted_rfr_y = rfr.predict(X)

        # Se añaden los datos predichos a la tabla.
        data["ExtraTreesRegressor"] = predicted_etr_y
        data["XGBRegressor"] = predicted_xgbm_y
        data["RandomForestRegressor"] = predicted_rfr_y

        dataArray.append(data)

        # Se añaden los datos del error a la tabla.
        data["ErrorExtraTreesRegressor"] = np.absolute(predicted_etr_y - data["Ocupacion"])
        data["ErrorXGBRegressor"] = np.absolute(predicted_xgbm_y - data["Ocupacion"])
        data["ErrorRandomForestRegressor"] = np.absolute(predicted_rfr_y - data["Ocupacion"])

        # Se generan los gráficos.
        fig, ax = plt.subplots(figsize=(10, 6))
        date_form = DateFormatter("%H:%M")
        ax.xaxis.set_major_formatter(date_form)
        plt.plot(data["Timestamp"], data["Ocupacion"], label="Ocupacion", color="red")
        plt.plot(data["Timestamp"], data["ExtraTreesRegressor"], label="ExtraTreesRegressor", color="blue")
        plt.plot(data["Timestamp"], data["XGBRegressor"], label="XGBRegressor", color="green")
        plt.plot(data["Timestamp"], data["RandomForestRegressor"], label="RandomForestRegressor", color="yellow")
        plt.xlabel("Hora")
        plt.ylabel("Ocupacion")
        plt.legend()
        plt.grid()
        plt.title(name)
        path1 = Path(folder, "prediction", name + ".jpg")
        plt.savefig(path1)
        plt.clf()
        plt.close()

        plt.plot(data["Timestamp"], data["Ocupacion"], label="Ocupacion", color="red")
        plt.plot(data["Timestamp"], data["ErrorExtraTreesRegressor"], label="ErrorExtraTreesRegressor", color="blue")
        plt.plot(data["Timestamp"], data["ErrorXGBRegressor"], label="ErrorXGBRegressor", color="green")
        plt.plot(data["Timestamp"], data["ErrorRandomForestRegressor"], label="ErrorRandomForestRegressor", color="yellow")
        plt.xlabel("Hora")
        plt.ylabel("Error")
        plt.legend()
        plt.grid()
        plt.title(name)
        path2 = Path(folder, "error", name + ".jpg")
        plt.savefig(path2)
        plt.clf()
        plt.close()

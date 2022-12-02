import numpy as np
import pandas as pd
import joblib
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
import warnings

warnings.filterwarnings("ignore")

testSet = pd.read_csv("../docs/filled-test-set.csv", sep=";")

est = joblib.load('../models/ExtraTreesRegressor.pkl')
xgbm = joblib.load('../models/XGBRegressor.pkl')
rfr = joblib.load('../models/RandomForestRegressor.pkl')

testSet["Timestamp"] = pd.to_datetime(testSet["Timestamp"])
dates = testSet["Timestamp"].dt.date.unique()
dataArray = []

for date in dates:
    group = testSet.loc[testSet["Timestamp"].dt.date == date]
    data = pd.DataFrame(group[["Timestamp", "Ocupacion"]], columns=["Timestamp", "Ocupacion"])

    name = date.strftime("%Y-%m-%d")

    X = group.loc[:, (group.columns != "Timestamp") & (group.columns != "Ocupacion")]

    predicted_etr_y = est.predict(X)
    predicted_xgbm_y = xgbm.predict(X)
    predicted_rfr_y = rfr.predict(X)

    data["ExtraTreesRegressor"] = predicted_etr_y
    data["XGBRegressor"] = predicted_xgbm_y
    data["RandomForestRegressor"] = predicted_rfr_y

    dataArray.append(data)

    data["ErrorExtraTreesRegressor"] = np.absolute(predicted_etr_y - data["Ocupacion"])
    data["ErrorXGBRegressor"] = np.absolute(predicted_xgbm_y - data["Ocupacion"])
    data["ErrorRandomForestRegressor"] = np.absolute(predicted_rfr_y - data["Ocupacion"])

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
    plt.savefig("../figuresPredict/prediction/" + name + ".jpg")
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
    plt.savefig("../figuresPredict/error/" + name + ".jpg")
    plt.clf()
    plt.close()



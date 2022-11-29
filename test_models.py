import pandas as pd
import joblib
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings("ignore")

testSet = pd.read_csv("../docs/filled-test-set.csv", sep=";")

est = joblib.load('../models/ExtraTreesRegressor.pkl')
lgbm = joblib.load('../models/LGBMRegressor.pkl')
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
    predicted_lgbm_y = lgbm.predict(X)
    predicted_rfr_y = rfr.predict(X)

    data["ExtraTreesRegressor"] = predicted_etr_y
    data["LGBMRegressor"] = predicted_lgbm_y
    data["RandomForestRegressor"] = predicted_rfr_y

    dataArray.append(data)

    plt.figure(figsize=(10, 6))
    plt.plot(data["Timestamp"], data["Ocupacion"], label="Ocupacion", color="red")
    plt.plot(data["Timestamp"], data["ExtraTreesRegressor"], label="ExtraTreesRegressor", color="blue")
    plt.plot(data["Timestamp"], data["LGBMRegressor"], label="LGBMRegressor", color="green")
    plt.plot(data["Timestamp"], data["RandomForestRegressor"], label="RandomForestRegressor", color="yellow")
    plt.xlabel("Timestamp")
    plt.ylabel("Ocupacion")
    plt.legend()
    plt.title(name)
    plt.savefig("../figuresPredict/" + name + ".jpg")
    plt.clf()
    plt.close()



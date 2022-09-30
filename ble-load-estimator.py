import pandas as pd
import joblib
from functions import savePlotDays
import warnings
warnings.filterwarnings("ignore")


trainingSet = pd.read_csv("docs/filled-training-set.csv", sep=";")

est = joblib.load('models/HistGradientBoostingRegressor.pkl')
lgbm = joblib.load('models/LGBMRegressor.pkl')
rfr = joblib.load('models/RandomForestRegressor.pkl')

trainingSet["Timestamp"] = pd.to_datetime(trainingSet["Timestamp"])
dates = trainingSet["Timestamp"].dt.date.unique()
dataArray = []
for date in dates:
    group = trainingSet.loc[trainingSet["Timestamp"].dt.date == date]
    data = pd.DataFrame(group[["Timestamp", "Person Count"]], columns=["Timestamp", "Person Count"])

    X = group.loc[:, (group.columns != "Timestamp") & (group.columns != "Person Count") &
                     (group.columns != "Date Range Index")]

    predicted_est_y = est.predict(X)
    predicted_lgbm_y = lgbm.predict(X)
    predicted_rfr_y = rfr.predict(X)

    data["HistGradientBoostingRegressor"] = predicted_est_y
    data["LGBMRegressor"] = predicted_lgbm_y
    data["RandomForestRegressor"] = predicted_rfr_y

    dataArray.append(data)


savePlotDays(dataArray, "figuresPredict/")

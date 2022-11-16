import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None


def getTrainingSetFormat(data, columns=None, sample=5):
    """Función que devuelve un Dataframe con las columnas incluidas en el argumento columns"""

    if columns is None:
        columns = ["N MAC RA", "N MAC RB", "N MAC RC", "N MAC RD", "N MAC RE", "N MAC RDE", "N MAC RCE", "N MAC RCDE",
                   "N MAC RBE", "N MAC MEN RA 10", "N MAC MEN RB 10", "N MAC MEN RC 10", "N MAC MEN RD 10",
                   "N MAC MEN RE 10", "N MAC INTERVALO ANTERIOR", "N MAC DOS INTERVALOS ANTERIORES"]

    dataCopy = data.copy()
    dataCopy["Timestamp"] = pd.to_datetime(dataCopy["Timestamp"])
    dates = dataCopy["Timestamp"].dt.date.unique()
    nDates = len(dates)
    date = dataCopy["Timestamp"].dt.date[0]
    date = date.strftime('%Y-%m-%d')
    group = dataCopy.loc[dataCopy["Timestamp"].dt.strftime('%Y-%m-%d') == date]
    dateRangeIndex = pd.Series(np.zeros(len(dataCopy)), name="Date Range Index")

    minutes = np.linspace(0, (len(group) - 1) * sample, len(group), dtype=int)
    minutes = np.tile(minutes, nDates)
    minutes = pd.Series(minutes, name="Minutes")

    trainingSet = pd.DataFrame(dataCopy[["Timestamp", "Person Count"]], columns=["Timestamp", "Person Count"])
    trainingSet = pd.concat([trainingSet, minutes], axis=1)
    trainingSet = pd.concat([trainingSet, dataCopy[columns]], axis=1)
    trainingSet = pd.concat([trainingSet, dateRangeIndex], axis=1)

    trainingSet.loc[trainingSet["Minutes"] <= minutes[int(len(group) / 3)], "Date Range Index"] = 1
    trainingSet.loc[trainingSet["Minutes"] > minutes[int(2 * len(group) / 3)], "Date Range Index"] = 3
    trainingSet.loc[trainingSet["Date Range Index"] == 0, "Date Range Index"] = 2

    trainingSet["Is NaN"] = pd.isna(trainingSet["N MAC RA"])
    trainingSet["crossing"] = (trainingSet["Is NaN"] != trainingSet["Is NaN"].shift()).cumsum()
    trainingSet["count"] = trainingSet.groupby(["Is NaN", "crossing"]).cumcount(ascending=False)
    trainingSet.loc[trainingSet["Is NaN"] == False, "count"] = 0
    trainingSet.drop(["crossing", "Is NaN"], axis=1, inplace=True)

    nanValues = pd.DataFrame(dataCopy["Timestamp"], columns=["Timestamp"])
    nanValues["Number of NaN"] = trainingSet.isna().sum(axis=1)

    print("Number of NaN per column")
    print(trainingSet.isna().sum())

    trainingSet["Timestamp"] = pd.to_datetime(trainingSet["Timestamp"])
    go = True
    while go:
        maxNaN = trainingSet["count"].max()
        if maxNaN > 5:
            index = trainingSet["count"].idxmax()
            trainingSet.drop(range(index, index+maxNaN+1), axis=0, inplace=True)
        else:
            go = False
            trainingSet.drop(["count"], axis=1, inplace=True)

    trainingSet.set_index("Timestamp", inplace=True)

    trainingSetFill = trainingSet.resample("5T").mean().interpolate()
    trainingSetFill = trainingSetFill.loc[trainingSet.index]

    return trainingSet, nanValues, trainingSetFill


rawTrainingSet = pd.read_csv("docs/training-set.csv", sep=";")

filterTrainingSet, filterNaNValues, filterTrainingSetFill = getTrainingSetFormat(rawTrainingSet)

filterTrainingSet.to_csv("docs/filter-training-set.csv", sep=";", na_rep="NaN")
filterNaNValues.to_csv("docs/nan-values-set.csv", sep=";", index=False)
filterTrainingSetFill.to_csv("docs/filled-training-set.csv", sep=";")

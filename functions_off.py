import numpy as np
import pandas as pd
import pathlib
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None


def setDateTimeLimits(data, values, day, isDf=True):
    """Función que devuelve un conjunto de datos con los límites de tiempo establecidos."""

    initDate = pd.to_datetime(day + " 07:00:00")
    endDate = pd.to_datetime(day + " 21:55:00")

    if isDf:
        data.reset_index(inplace=True)
        initValues = [initDate] + values
        endValues = [endDate] + values
        dfInit = pd.DataFrame([initValues], columns=data.columns)
        dfEnd = pd.DataFrame([endValues], columns=data.columns)
        if initDate not in data["Timestamp"].unique():
            data = pd.concat([dfInit, data])

        if endDate not in data["Timestamp"].unique():
            data = pd.concat([data, dfEnd])

    else:
        dfInit = pd.Series(values, index=[initDate], name="Timestamp")
        dfEnd = pd.Series(values, index=[endDate], name="Timestamp")
        if initDate not in data:
            data = pd.concat([dfInit, data])

        if endDate not in data:
            data = pd.concat([data, dfEnd])

    return data


def readDataFromDirectory(dataPath, personCountPath, statePath):
    """Función que lee los archivos de datos de los receptores Bluetooth y del contador de personas y los concentra en un array"""

    dataArray = []
    personCountArray = []
    stateArray = []
    dataPath = pathlib.Path(dataPath)
    personCountPath = pathlib.Path(personCountPath)
    statePath = pathlib.Path(statePath)
    for file in dataPath.iterdir():
        data = pd.read_csv(file, sep=';', usecols=["Timestamp int.", "Raspberry", "Nº Mensajes", "MAC"])
        data["Timestamp int."] = pd.to_datetime(data["Timestamp int."], dayfirst=True)
        data = data.rename(columns={"Timestamp int.": "Timestamp"})
        data = data.drop(data[data["MAC"] == "00:00:00:00:00:00"].index).reset_index(drop=True)
        dataArray.append(data)

    for file in personCountPath.iterdir():
        personCount = pd.read_csv(file, sep=';', usecols=["Fecha", "Hora", "Ocupacion", "Estado"])
        personCount.insert(0, "Timestamp", personCount.Fecha.str.cat(personCount.Hora, sep=" "))
        personCount.drop(columns=["Fecha", "Hora"], inplace=True)
        personCount["Timestamp"] = pd.to_datetime(personCount["Timestamp"])
        personCount = personCount.groupby(pd.Grouper(key="Timestamp", freq="5T")).last()
        day = personCount.index.date[0].strftime(format="%Y-%m-%d")

        personCount = setDateTimeLimits(personCount, [0, 1], day)

        personCount.set_index("Timestamp", inplace=True)
        personCount = personCount.resample("5T").last()
        personCount["Estado"].fillna(1, inplace=True)
        personCount["Ocupacion"].fillna(0, inplace=True)
        personCount.loc[personCount["Estado"] == 0, "Ocupacion"] = np.nan
        personCountArray.append(personCount)

    for file in statePath.iterdir():
        state = pd.read_csv(file, sep=';')
        state.insert(0, "Timestamp", state.Fecha.str.cat(state.Hora, sep=" "))
        state.drop(columns=["Fecha", "Hora", "Indice intervalo"], inplace=True)
        state["Timestamp"] = pd.to_datetime(state["Timestamp"])
        state.set_index("Timestamp", inplace=True)
        stateArray.append(state)

    return dataArray, personCountArray, stateArray


def parseDataByRaspberry(data):
    """Función que devuelve un conjunto de datos filtrado por cada Raspberry. Devuelve un conjunto por Raspberry."""

    dataCopy = data.copy()
    dataInterval1 = dataCopy.loc[dataCopy['Raspberry'] == 'Raspberry A']
    dataInterval2 = dataCopy.loc[dataCopy['Raspberry'] == 'Raspberry B']
    dataInterval3 = dataCopy.loc[dataCopy['Raspberry'] == 'Raspberry C']
    dataInterval4 = dataCopy.loc[dataCopy['Raspberry'] == 'Raspberry D']
    dataInterval5 = dataCopy.loc[dataCopy['Raspberry'] == 'Raspberry E']

    return dataInterval1, dataInterval2, dataInterval3, dataInterval4, dataInterval5


def parseDataByRaspberryTime(data):
    """Función que devuelve conjuntos de datos con valores únicos filtrados por Raspberry y agrupados por Timestamp."""

    dataInterval1, dataInterval2, dataInterval3, dataInterval4, dataInterval5 = parseDataByRaspberry(data)

    dataInterval1 = dataInterval1.groupby('Timestamp').nunique()
    dataInterval2 = dataInterval2.groupby('Timestamp').nunique()
    dataInterval3 = dataInterval3.groupby('Timestamp').nunique()
    dataInterval4 = dataInterval4.groupby('Timestamp').nunique()
    dataInterval5 = dataInterval5.groupby('Timestamp').nunique()

    return dataInterval1, dataInterval2, dataInterval3, dataInterval4, dataInterval5


def getTotalDevicesByRaspberry(data, state):
    """Función que devuelve conjuntos de datos con el número de dispositivos únicos filtrados por Raspberry y agrupados
    por Timestamp."""

    dataInterval1, dataInterval2, dataInterval3, dataInterval4, dataInterval5 = parseDataByRaspberryTime(data)
    RADownInterval, RBDownInterval, RCDownInterval, RDDownInterval, REDownInterval = state

    dataInterval1.drop(columns=["Nº Mensajes", "Raspberry"], inplace=True)
    dataInterval2.drop(columns=["Nº Mensajes", "Raspberry"], inplace=True)
    dataInterval3.drop(columns=["Nº Mensajes", "Raspberry"], inplace=True)
    dataInterval4.drop(columns=["Nº Mensajes", "Raspberry"], inplace=True)
    dataInterval5.drop(columns=["Nº Mensajes", "Raspberry"], inplace=True)

    dataArray = [dataInterval1, dataInterval2, dataInterval3, dataInterval4, dataInterval5]
    statusList = [RADownInterval, RBDownInterval, RCDownInterval, RDDownInterval, REDownInterval]
    finalDataList = []

    day = data["Timestamp"].dt.date[0].strftime(format="%Y-%m-%d")

    for i in range(len(dataArray)):
        column = dataArray[i]
        column = setDateTimeLimits(column, [0], day)
        column.set_index("Timestamp", inplace=True)
        column = column.resample("5T").asfreq().fillna(0)

        column.loc[statusList[i], "MAC"] = np.nan

        finalDataList.append(column)

    totalMACRA, totalMACRB, totalMACRC, totalMACRD, totalMACRE = finalDataList

    totalMACRA = totalMACRA["MAC"].values
    totalMACRB = totalMACRB["MAC"].values
    totalMACRC = totalMACRC["MAC"].values
    totalMACRD = totalMACRD["MAC"].values
    totalMACRE = totalMACRE["MAC"].values

    return totalMACRA, totalMACRB, totalMACRC, totalMACRD, totalMACRE


def getTotalDevicesByPairRaspberries(data, state):
    """Función que devuelve cuatro listas compuestas por los dispositivos captados en el mismo intervalo de tiempo por
    las parejas C-E, D-E, B-E y el trío C-D-E"""

    dataInterval1, dataInterval2, dataInterval3, dataInterval4, dataInterval5 = parseDataByRaspberry(data)
    RADownInterval, RBDownInterval, RCDownInterval, RDDownInterval, REDownInterval = state

    dataInterval1.drop(columns="Nº Mensajes", inplace=True)
    dataInterval2.drop(columns="Nº Mensajes", inplace=True)
    dataInterval3.drop(columns="Nº Mensajes", inplace=True)
    dataInterval4.drop(columns="Nº Mensajes", inplace=True)
    dataInterval5.drop(columns="Nº Mensajes", inplace=True)

    nDevicesIntervalDataRAMerge = dataInterval1.set_index("Timestamp")
    nDevicesIntervalDataRBMerge = dataInterval2.set_index("Timestamp")
    nDevicesIntervalDataRCMerge = dataInterval3.set_index("Timestamp")
    nDevicesIntervalDataRDMerge = dataInterval4.set_index("Timestamp")
    nDevicesIntervalDataREMerge = dataInterval5.set_index("Timestamp")

    nDevicesIntervalDataRDEMerge = nDevicesIntervalDataRDMerge.merge(nDevicesIntervalDataREMerge, how='outer',
                                                                     on=("Timestamp", "MAC"), copy=False,
                                                                     suffixes=("_d", "_e"))
    nDevicesIntervalDataRCDEMerge = nDevicesIntervalDataRDEMerge.merge(nDevicesIntervalDataRCMerge, how='outer',
                                                                       on=("Timestamp", "MAC"), copy=False)
    nDevicesIntervalDataRBCDEMerge = nDevicesIntervalDataRCDEMerge.merge(nDevicesIntervalDataRBMerge, how='outer',
                                                                         on=("Timestamp", "MAC"), copy=False,
                                                                         suffixes=("_c", "_b"))
    nDevicesIntervalDataRABCDEMerge = nDevicesIntervalDataRBCDEMerge.merge(nDevicesIntervalDataRAMerge, how='outer',
                                                                           on=("Timestamp", "MAC"), copy=False)

    group = nDevicesIntervalDataRABCDEMerge.groupby(["Timestamp", "MAC"]).nunique()

    group_CDE = group.loc[(group["Raspberry_c"] == 1) & (group["Raspberry_d"] == 1) & (group["Raspberry_e"] == 1)]
    group_CE = group.loc[(group["Raspberry_c"] == 1) & (group["Raspberry_e"] == 1)]
    group_DE = group.loc[(group["Raspberry_d"] == 1) & (group["Raspberry_e"] == 1)]
    group_BE = group.loc[(group["Raspberry_b"] == 1) & (group["Raspberry_e"] == 1)]

    day = group.index.get_level_values(0).date[0].strftime(format="%Y-%m-%d")

    dataArray = [group_CDE, group_CE, group_DE, group_BE]
    finalDataList = []

    for i in range(len(dataArray)):
        column = dataArray[i]
        column.reset_index(inplace=True)
        column = column["Timestamp"].value_counts(sort=False)
        column = setDateTimeLimits(column, 0, day, False)
        column = column.resample("5T").asfreq().fillna(0)

        if i == 0:
            column.loc[RCDownInterval] = np.nan
            column.loc[RDDownInterval] = np.nan
            column.loc[REDownInterval] = np.nan
        elif i == 1:
            column.loc[RCDownInterval] = np.nan
            column.loc[REDownInterval] = np.nan
        elif i == 2:
            column.loc[RDDownInterval] = np.nan
            column.loc[REDownInterval] = np.nan
        else:
            column.loc[RBDownInterval] = np.nan
            column.loc[REDownInterval] = np.nan

        finalDataList.append(column)

    totalMACRCDE, totalMACRCE, totalMACRDE, totalMACRBE = finalDataList

    return totalMACRCDE.values, totalMACRCE.values, totalMACRDE.values, totalMACRBE.values


def getTotalDeviceByMessageNumber(data, state):
    """Función que devuelve tres listas por Raspberry, una por intervalo de número de mensajes por debajo
    de 10, entre 10 y 30 y superior a 30."""

    dataInterval1, dataInterval2, dataInterval3, dataInterval4, dataInterval5 = parseDataByRaspberry(data)
    RADownInterval, RBDownInterval, RCDownInterval, RDDownInterval, REDownInterval = state

    dataInterval1 = dataInterval1.set_index("Timestamp")
    dataInterval2 = dataInterval2.set_index("Timestamp")
    dataInterval3 = dataInterval3.set_index("Timestamp")
    dataInterval4 = dataInterval4.set_index("Timestamp")
    dataInterval5 = dataInterval5.set_index("Timestamp")

    dataInterval1 = dataInterval1.groupby(["Timestamp", "MAC"]).sum()
    dataInterval2 = dataInterval2.groupby(["Timestamp", "MAC"]).sum()
    dataInterval3 = dataInterval3.groupby(["Timestamp", "MAC"]).sum()
    dataInterval4 = dataInterval4.groupby(["Timestamp", "MAC"]).sum()
    dataInterval5 = dataInterval5.groupby(["Timestamp", "MAC"]).sum()

    totalMACRA_10 = dataInterval1.loc[dataInterval1["Nº Mensajes"] <= 10]
    totalMACRA_1030 = dataInterval1.loc[(dataInterval1["Nº Mensajes"] > 10) & (dataInterval1["Nº Mensajes"] <= 30)]
    totalMACRA_30 = dataInterval1.loc[dataInterval1["Nº Mensajes"] > 30]

    totalMACRB_10 = dataInterval2.loc[dataInterval2["Nº Mensajes"] <= 10]
    totalMACRB_1030 = dataInterval2.loc[(dataInterval2["Nº Mensajes"] > 10) & (dataInterval2["Nº Mensajes"] <= 30)]
    totalMACRB_30 = dataInterval2.loc[dataInterval2["Nº Mensajes"] > 30]

    totalMACRC_10 = dataInterval3.loc[dataInterval3["Nº Mensajes"] <= 10]
    totalMACRC_1030 = dataInterval3.loc[(dataInterval3["Nº Mensajes"] > 10) & (dataInterval3["Nº Mensajes"] <= 30)]
    totalMACRC_30 = dataInterval3.loc[dataInterval3["Nº Mensajes"] > 30]

    totalMACRD_10 = dataInterval4.loc[dataInterval4["Nº Mensajes"] <= 10]
    totalMACRD_1030 = dataInterval4.loc[(dataInterval4["Nº Mensajes"] > 10) & (dataInterval4["Nº Mensajes"] <= 30)]
    totalMACRD_30 = dataInterval4.loc[dataInterval4["Nº Mensajes"] > 30]

    totalMACRE_10 = dataInterval5.loc[dataInterval5["Nº Mensajes"] <= 10]
    totalMACRE_1030 = dataInterval5.loc[(dataInterval5["Nº Mensajes"] > 10) & (dataInterval5["Nº Mensajes"] <= 30)]
    totalMACRE_30 = dataInterval5.loc[dataInterval5["Nº Mensajes"] > 30]

    day = data["Timestamp"].dt.date[0].strftime(format="%Y-%m-%d")

    dataArray = [totalMACRA_10, totalMACRA_1030, totalMACRA_30, totalMACRB_10, totalMACRB_1030, totalMACRB_30,
                 totalMACRC_10,
                 totalMACRC_1030, totalMACRC_30, totalMACRD_10, totalMACRD_1030, totalMACRD_30, totalMACRE_10,
                 totalMACRE_1030, totalMACRE_30]
    finalDataList = []

    for i in range(len(dataArray)):
        column = dataArray[i]
        column.reset_index(inplace=True)
        column = column["Timestamp"].value_counts(sort=False)
        column = setDateTimeLimits(column, 0, day, False)
        column = column.resample("5T").asfreq().fillna(0)

        if i < 3:
            column.loc[RADownInterval] = np.nan
        elif i < 6:
            column.loc[RBDownInterval] = np.nan
        elif i < 9:
            column.loc[RCDownInterval] = np.nan
        elif i < 12:
            column.loc[RDDownInterval] = np.nan
        else:
            column.loc[REDownInterval] = np.nan

        finalDataList.append(column.values)

    totalMACRA_10, totalMACRA_1030, totalMACRA_30, totalMACRB_10, totalMACRB_1030, totalMACRB_30, totalMACRC_10, \
    totalMACRC_1030, totalMACRC_30, totalMACRD_10, totalMACRD_1030, totalMACRD_30, totalMACRE_10, \
    totalMACRE_1030, totalMACRE_30 = finalDataList

    return totalMACRA_10, totalMACRA_1030, totalMACRA_30, totalMACRB_10, totalMACRB_1030, totalMACRB_30, totalMACRC_10, \
           totalMACRC_1030, totalMACRC_30, totalMACRD_10, totalMACRD_1030, totalMACRD_30, totalMACRE_10, \
           totalMACRE_1030, totalMACRE_30


def getTotalDevicesInPreviousInterval(data, state):
    """Función que devuelve una lista con el número de dispositivos registrados en el intervalo de tiempo actual y el
    anterior."""

    dataCopy = data.copy()
    RADownInterval, RBDownInterval, RCDownInterval, RDDownInterval, REDownInterval = state

    dataCopy[(dataCopy["Timestamp"].isin(RADownInterval)) & (dataCopy["Raspberry"].isin(["Raspberry A"]))] = np.nan
    dataCopy[(dataCopy["Timestamp"].isin(RBDownInterval)) & (dataCopy["Raspberry"].isin(["Raspberry B"]))] = np.nan
    dataCopy[(dataCopy["Timestamp"].isin(RCDownInterval)) & (dataCopy["Raspberry"].isin(["Raspberry C"]))] = np.nan
    dataCopy[(dataCopy["Timestamp"].isin(RDDownInterval)) & (dataCopy["Raspberry"].isin(["Raspberry D"]))] = np.nan
    dataCopy[(dataCopy["Timestamp"].isin(REDownInterval)) & (dataCopy["Raspberry"].isin(["Raspberry E"]))] = np.nan

    day = dataCopy["Timestamp"].iloc[0].date().strftime('%Y-%m-%d')
    dataCopy.set_index("Timestamp", inplace=True)
    dataCopy.dropna(inplace=True)
    dataCopy.drop(columns=["Nº Mensajes", "Raspberry"], inplace=True)

    timestampSplit = []
    for date in dataCopy.index.unique():
        if len(dataCopy.loc[date]) == 1:
            timestampSplit.append(np.array([dataCopy.loc[date]["MAC"]], dtype=object))
        else:
            timestampSplit.append(dataCopy.loc[date]["MAC"].unique())

    totalMACPreviousInterval = pd.DataFrame([[pd.to_datetime(day + " 07:00:00"), 0]], columns=["Timestamp", "MAC"])
    for i in range(1, len(timestampSplit)):
        date = dataCopy.index.unique()[i]
        coincidences = len(set(timestampSplit[i]) & set(timestampSplit[i - 1]))
        actualDf = pd.DataFrame([[date, coincidences]], columns=["Timestamp", "MAC"])
        totalMACPreviousInterval = pd.concat([totalMACPreviousInterval, actualDf])

    totalMACPreviousInterval.set_index("Timestamp", inplace=True)
    totalMACPreviousInterval = totalMACPreviousInterval.resample("5T").asfreq().fillna(0)
    totalMACPreviousInterval = np.array(totalMACPreviousInterval["MAC"].values)

    return totalMACPreviousInterval


def getTotalDevicesInTwoPreviousIntervals(data, state):
    """Función que devuelve una lista con el número de dispositivos registrados en el intervalo de tiempo actual y los
    dos anteriores."""

    dataCopy = data.copy()
    RADownInterval, RBDownInterval, RCDownInterval, RDDownInterval, REDownInterval = state

    dataCopy[(dataCopy["Timestamp"].isin(RADownInterval)) & (dataCopy["Raspberry"].isin(["Raspberry A"]))] = np.nan
    dataCopy[(dataCopy["Timestamp"].isin(RBDownInterval)) & (dataCopy["Raspberry"].isin(["Raspberry B"]))] = np.nan
    dataCopy[(dataCopy["Timestamp"].isin(RCDownInterval)) & (dataCopy["Raspberry"].isin(["Raspberry C"]))] = np.nan
    dataCopy[(dataCopy["Timestamp"].isin(RDDownInterval)) & (dataCopy["Raspberry"].isin(["Raspberry D"]))] = np.nan
    dataCopy[(dataCopy["Timestamp"].isin(REDownInterval)) & (dataCopy["Raspberry"].isin(["Raspberry E"]))] = np.nan

    day = dataCopy["Timestamp"].iloc[0].date().strftime('%Y-%m-%d')
    dataCopy.set_index("Timestamp", inplace=True)
    dataCopy.dropna(inplace=True)
    dataCopy.drop(columns=["Nº Mensajes", "Raspberry"], inplace=True)

    timestampSplit = []
    for date in dataCopy.index.unique():
        if len(dataCopy.loc[date]) == 1:
            timestampSplit.append(np.array([dataCopy.loc[date]["MAC"]], dtype=object))
        else:
            timestampSplit.append(dataCopy.loc[date]["MAC"].unique())

    totalMACTwoPreviousInterval = pd.DataFrame(
        [[pd.to_datetime(day + " 07:00:00"), 0], [pd.to_datetime(day + " 07:05:00"), 0]], columns=["Timestamp", "MAC"])
    for i in range(2, len(timestampSplit)):
        date = dataCopy.index.unique()[i]
        coincidences = len(set(timestampSplit[i]) & set(timestampSplit[i - 1]) & set(timestampSplit[i - 2]))
        actualDf = pd.DataFrame([[date, coincidences]], columns=["Timestamp", "MAC"])
        totalMACTwoPreviousInterval = pd.concat([totalMACTwoPreviousInterval, actualDf])

    totalMACTwoPreviousInterval.set_index("Timestamp", inplace=True)
    totalMACTwoPreviousInterval = totalMACTwoPreviousInterval.resample("5T").asfreq().fillna(0)
    totalMACTwoPreviousInterval = np.array(totalMACTwoPreviousInterval["MAC"].values)

    return totalMACTwoPreviousInterval


def savePlotColumns(data, path, path2):
    """Función que guarda en una carpeta las gráficas para cada una de las columnas del training set."""

    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    day = data["Timestamp"].iloc[0].date().strftime('%Y-%m-%d')

    for i in range(3, len(data.columns)):
        name = data.columns[i] + "_" + day
        nameDate = day + "_" + data.columns[i]

        plt.figure(figsize=(10, 6))
        plt.plot(data["Timestamp"], data["Ocupacion"], label="Ocupacion", color="red")
        plt.plot(data["Timestamp"], data[data.columns[i]], label=data.columns[i], color="blue")
        plt.xlabel("Timestamp")
        plt.ylabel("Devices")
        plt.legend()
        plt.title(name)
        plt.savefig(path + name + '.jpg')
        plt.title(nameDate)
        plt.savefig(path2 + nameDate + '.jpg')
        plt.clf()
        plt.close()


def fillTrainingSet(data):
    """Función que completa el conjunto de entrenamiento filtrado rellenando los valores nulos o eliminando las filas
    imposibles de interpolar"""

    dataCopy = data.copy()
    dataCopy.set_index("Timestamp", inplace=True)
    filledSet = dataCopy.resample("5T").mean().interpolate()
    filledSet.reset_index(inplace=True)
    filledSet = filledSet.round(3)
    filledSet = filledSet[filledSet["Ocupacion"].notna()]
    return filledSet


def getTrainingDataset(dataArray, personCountArray, stateArray):
    """Función que devuelve un conjunto de datos para el algoritmo de Machine Learning y un dataframe con los valores
    acumulados hasta ese momento"""

    columns = ["Timestamp", "Ocupacion", "Minutes", "N MAC TOTAL", "N MAC RA", "N MAC RB", "N MAC RC", "N MAC RD",
               "N MAC RE",
               "N MAC RDE", "N MAC RCE", "N MAC RCDE", "N MAC RBE", "N MAC MEN RA 10", "N MAC MEN RA 10-30",
               "N MAC MEN RA 30", "N MAC MEN RB 10", "N MAC MEN RB 10-30", "N MAC MEN RB 30", "N MAC MEN RC 10",
               "N MAC MEN RC 10-30", "N MAC MEN RC 30", "N MAC MEN RD 10", "N MAC MEN RD 10-30", "N MAC MEN RD 30",
               "N MAC MEN RE 10", "N MAC MEN RE 10-30", "N MAC MEN RE 30", "N MAC INTERVALO ANTERIOR",
               "N MAC DOS INTERVALOS ANTERIORES"]

    columnsFilter = ["Timestamp", "Ocupacion", "Minutes", "N MAC RA", "N MAC RB", "N MAC RC", "N MAC RD", "N MAC RE",
                     "N MAC RDE", "N MAC RCE", "N MAC RCDE",
                     "N MAC RBE", "N MAC MEN RA 10", "N MAC MEN RB 10", "N MAC MEN RC 10", "N MAC MEN RD 10",
                     "N MAC MEN RE 10", "N MAC INTERVALO ANTERIOR", "N MAC DOS INTERVALOS ANTERIORES"]

    trainingDataSet = pd.DataFrame(columns=columns)
    filterDataSet = pd.DataFrame(columns=columnsFilter)
    filledDataSet = pd.DataFrame(columns=columnsFilter)
    sample = 5
    length = len(dataArray[0]["Timestamp"].unique())
    minutes = np.linspace(0, (length - 1) * sample, length, dtype=int)

    for i in range(len(dataArray)):
        data = dataArray[i]
        personCount = personCountArray[i]
        state = stateArray[i]

        day = data["Timestamp"][0].date().strftime('%Y-%m-%d')

        RADownInterval = state.loc[state["RA(1/0)"] == 0].index
        RBDownInterval = state.loc[state["RB(1/0)"] == 0].index
        RCDownInterval = state.loc[state["RC(1/0)"] == 0].index
        RDDownInterval = state.loc[state["RD(1/0)"] == 0].index
        REDownInterval = state.loc[state["RE(1/0)"] == 0].index
        RDownInterval = (RADownInterval, RBDownInterval, RCDownInterval, RDDownInterval, REDownInterval)

        dataGroup = data.groupby("Timestamp").nunique()
        dataGroup = setDateTimeLimits(dataGroup, [np.nan, np.nan, np.nan], day)
        dataGroup.set_index("Timestamp", inplace=True)
        dataGroup = dataGroup.resample("5T").asfreq()
        totalMAC = dataGroup["MAC"].values

        totalMACRA, totalMACRB, totalMACRC, totalMACRD, totalMACRE = getTotalDevicesByRaspberry(data, RDownInterval)

        totalMACRCDE, totalMACRCE, totalMACRDE, totalMACRBE = getTotalDevicesByPairRaspberries(data, RDownInterval)

        totalMACRA_10, totalMACRA_1030, totalMACRA_30, totalMACRB_10, totalMACRB_1030, totalMACRB_30, totalMACRC_10, \
        totalMACRC_1030, totalMACRC_30, totalMACRD_10, totalMACRD_1030, totalMACRD_30, totalMACRE_10, totalMACRE_1030, \
        totalMACRE_30 = getTotalDeviceByMessageNumber(data, RDownInterval)

        totalMACPreviousInterval = getTotalDevicesInPreviousInterval(data, RDownInterval)

        totalMACTwoPreviousInterval = getTotalDevicesInTwoPreviousIntervals(data, RDownInterval)

        timestamp = data["Timestamp"].unique()
        timestamp = pd.Series(np.zeros(len(timestamp)), index=timestamp, name="Timestamp")
        timestamp = setDateTimeLimits(timestamp, 0, day, False)
        timestamp = timestamp.resample("5T").asfreq()
        timestamp = timestamp.index.strftime('%Y-%m-%d %H:%M:%S')

        trainingData = np.array(np.transpose([timestamp, personCount["Ocupacion"].values, minutes, totalMAC,
                                              totalMACRA,
                                              totalMACRB,
                                              totalMACRC, totalMACRD, totalMACRE, totalMACRDE, totalMACRCE,
                                              totalMACRCDE, totalMACRBE,
                                              totalMACRA_10,
                                              totalMACRA_1030, totalMACRA_30, totalMACRB_10, totalMACRB_1030,
                                              totalMACRB_30, totalMACRC_10,
                                              totalMACRC_1030, totalMACRC_30, totalMACRD_10, totalMACRD_1030,
                                              totalMACRD_30, totalMACRE_10,
                                              totalMACRE_1030, totalMACRE_30, totalMACPreviousInterval,
                                              totalMACTwoPreviousInterval]))

        trainingSet = pd.DataFrame(trainingData, columns=columns)

        trainingDataSet = pd.concat([trainingDataSet, trainingSet], ignore_index=True)

        print("Guardando gráficas de los datos calculados de la fecha " + day + "...")
        savePlotColumns(trainingSet, "../figures/", "../figuresDate/")

        filterSet = trainingSet[columnsFilter]
        filterDataSet = pd.concat([filterDataSet, filterSet], ignore_index=True)

        filledSet = fillTrainingSet(filterSet)
        filledDataSet = pd.concat([filledDataSet, filledSet], ignore_index=True)

        print("Guardando gráficas de los datos limpios de la fecha " + day + "...")
        savePlotColumns(filledSet, "../figuresFilled/", "../figuresDateFilled/")

    trainingDataSet.to_csv("../docs/training-set.csv", sep=";", na_rep="NaN", index=False)
    filterDataSet.to_csv("../docs/filter-training-set.csv", sep=";", na_rep="NaN", index=False)
    filledDataSet.to_csv("../docs/filled-training-set.csv", sep=";", na_rep="NaN", index=False)

    return trainingDataSet, filterDataSet, filledDataSet


dataList, personCountList, stateList = readDataFromDirectory("../docs/data", "../docs/personcount", "../docs/state")

getTrainingDataset(dataList, personCountList, stateList)
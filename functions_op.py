import numpy as np
import pandas as pd
import pathlib
import os


def readDataFromDirectory(dataPath, personCountPath, statePath):
    """Función que lee los archivos de datos de los receptores Bluetooth y del contador de personas y los concentra en un array"""

    dataArray = []
    personCountArray = []
    stateArray = []
    dates = []
    dataPath = pathlib.Path(dataPath)
    personCountPath = pathlib.Path(personCountPath)
    statePath = pathlib.Path(statePath)
    for file in dataPath.iterdir():
        data = pd.read_csv(file, sep=';', usecols=["Timestamp int.", "Raspberry", "Nº Mensajes", "MAC"])
        data["Timestamp int."] = pd.to_datetime(data["Timestamp int."], dayfirst=True)
        data = data.rename(columns={"Timestamp int.": "Timestamp"})
        dataArray.append(data)
        dates.append(data["Timestamp"].dt.date[0])

    for file in personCountPath.iterdir():
        personCount = pd.read_csv(file, sep=';', usecols=["Fecha", "Hora", "Ocupacion", "Estado"])
        personCount.insert(0, "Timestamp", personCount.Fecha.str.cat(personCount.Hora, sep=" "))
        personCount.drop(columns=["Fecha", "Hora"], inplace=True)
        personCount["Timestamp"] = pd.to_datetime(personCount["Timestamp"])
        personCount.set_index("Timestamp", inplace=True)
        personCount = personCount.resample("5T").last()
        personCount.loc[personCount["Estado"] == 0, "Ocupacion"] = np.nan
        personCountArray.append(personCount)

    for file in statePath.iterdir():
        state = pd.read_csv(file, sep=';')
        state.insert(0, "Timestamp", state.Fecha.str.cat(state.Hora, sep=" "))
        state.drop(columns=["Fecha", "Hora", "Indice intervalo"], inplace=True)
        state["Timestamp"] = pd.to_datetime(state["Timestamp"])
        state.set_index("Timestamp", inplace=True)
        stateArray.append(state)

    return dataArray, personCountArray, stateArray, dates


def parseDataByRaspberry(data):
    """Función que devuelve un conjunto de datos filtrado por cada Raspberry. Devuelve un conjunto por Raspberry."""

    dataInterval1 = data.loc[data['Raspberry'] == 'Raspberry A']
    dataInterval2 = data.loc[data['Raspberry'] == 'Raspberry B']
    dataInterval3 = data.loc[data['Raspberry'] == 'Raspberry C']
    dataInterval4 = data.loc[data['Raspberry'] == 'Raspberry D']
    dataInterval5 = data.loc[data['Raspberry'] == 'Raspberry E']

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

    dataInterval1.loc[RADownInterval, "MAC"] = np.nan
    dataInterval2.loc[RBDownInterval, "MAC"] = np.nan
    dataInterval3.loc[RCDownInterval, "MAC"] = np.nan
    dataInterval4.loc[RDDownInterval, "MAC"] = np.nan
    dataInterval5.loc[REDownInterval, "MAC"] = np.nan

    totalMACRA = dataInterval1["MAC"].values
    totalMACRB = dataInterval2["MAC"].values
    totalMACRC = dataInterval3["MAC"].values
    totalMACRD = dataInterval4["MAC"].values
    totalMACRE = dataInterval5["MAC"].values

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

    nDevicesIntervalDataRAMerge.loc[RADownInterval, "MAC"] = np.nan
    nDevicesIntervalDataRBMerge.loc[RBDownInterval, "MAC"] = np.nan
    nDevicesIntervalDataRCMerge.loc[RCDownInterval, "MAC"] = np.nan
    nDevicesIntervalDataRDMerge.loc[RDDownInterval, "MAC"] = np.nan
    nDevicesIntervalDataREMerge.loc[REDownInterval, "MAC"] = np.nan

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

    group_CDE = group.loc[(group["Raspberry"] == 0) & (group["Raspberry_b"] == 0) & (group["Raspberry_c"] == 1) & (
            group["Raspberry_d"] == 1) & (group["Raspberry_e"] == 1)]
    group_CE = group.loc[(group["Raspberry"] == 0) & (group["Raspberry_b"] == 0) & (group["Raspberry_c"] == 1) & (
            group["Raspberry_d"] == 0) & (group["Raspberry_e"] == 1)]
    group_DE = group.loc[(group["Raspberry"] == 0) & (group["Raspberry_b"] == 0) & (group["Raspberry_c"] == 0) & (
            group["Raspberry_d"] == 1) & (group["Raspberry_e"] == 1)]
    group_BE = group.loc[(group["Raspberry"] == 0) & (group["Raspberry_b"] == 1) & (group["Raspberry_c"] == 0) & (
            group["Raspberry_d"] == 0) & (group["Raspberry_e"] == 1)]

    # Por cada MAC en cada intervalo de tiempo, existe un valor de la columna Timestamp, es decir, si hay 4 MAC en un intervalo,
    # Se generarán 4 filas con el mismo valor de Timestamp. Si cuento las veces que se repite cada intervalo, sé cuantas MAC hay.

    day = group_CDE.index.get_level_values(0).date[0].strftime(format="%Y-%m-%d")
    initDate = pd.to_datetime(day + " 07:00:00")
    endDate = pd.to_datetime(day + " 21:55:00")

    dataList = [group_CDE, group_CE, group_DE, group_BE]
    finalDataList = []

    for column in dataList:
        column.reset_index(inplace=True)
        column = column["Timestamp"].value_counts(sort=False)

        if initDate not in column:
            column = pd.concat([pd.Series(np.nan, index=[initDate], name="Timestamp"), column])

        if endDate not in column:
            column = pd.concat([column, pd.Series(np.nan, index=[endDate], name="Timestamp")])

        finalDataList.append(column.resample("5T").last().values)

    finalDataList = tuple(finalDataList)
    totalMACRCDE, totalMACRCE, totalMACRDE, totalMACRBE = finalDataList

    return totalMACRCDE, totalMACRCE, totalMACRDE, totalMACRBE


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

    dataInterval1.loc[RADownInterval, "MAC"] = np.nan
    dataInterval2.loc[RBDownInterval, "MAC"] = np.nan
    dataInterval3.loc[RCDownInterval, "MAC"] = np.nan
    dataInterval4.loc[RDDownInterval, "MAC"] = np.nan
    dataInterval5.loc[REDownInterval, "MAC"] = np.nan

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

    day = dataInterval1.index.get_level_values(0).date[0].strftime(format="%Y-%m-%d")
    initDate = pd.to_datetime(day + " 07:00:00")
    endDate = pd.to_datetime(day + " 21:55:00")

    dataList = [totalMACRA_10, totalMACRA_1030, totalMACRA_30, totalMACRB_10, totalMACRB_1030, totalMACRB_30, totalMACRC_10,
                totalMACRC_1030, totalMACRC_30, totalMACRD_10, totalMACRD_1030, totalMACRD_30, totalMACRE_10,
                totalMACRE_1030, totalMACRE_30]
    finalDataList = []

    for column in dataList:
        column.reset_index(inplace=True)
        column = column["Timestamp"].value_counts(sort=False)

        if initDate not in column:
            column = pd.concat([pd.Series(np.nan, index=[initDate], name="Timestamp"), column])

        if endDate not in column:
            column = pd.concat([column, pd.Series(np.nan, index=[endDate], name="Timestamp")])

        finalDataList.append(column.resample("5T").last().values)

    finalDataList = tuple(finalDataList)
    totalMACRA_10, totalMACRA_1030, totalMACRA_30, totalMACRB_10, totalMACRB_1030, totalMACRB_30, totalMACRC_10, \
    totalMACRC_1030, totalMACRC_30, totalMACRD_10, totalMACRD_1030, totalMACRD_30, totalMACRE_10, \
    totalMACRE_1030, totalMACRE_30 = finalDataList

    return totalMACRA_10, totalMACRA_1030, totalMACRA_30, totalMACRB_10, totalMACRB_1030, totalMACRB_30, totalMACRC_10, \
           totalMACRC_1030, totalMACRC_30, totalMACRD_10, totalMACRD_1030, totalMACRD_30, totalMACRE_10, \
           totalMACRE_1030, totalMACRE_30


def getTotalDevicesInPreviousInterval(data, state):
    """Función que devuelve una lista con el número de dispositivos registrados en el intervalo de tiempo actual y el
    anterior."""

    RADownInterval, RBDownInterval, RCDownInterval, RDDownInterval, REDownInterval = state

    data[(data["Timestamp"].isin(RADownInterval)) & (data["Raspberry"].isin(["Raspberry A"]))] = np.nan
    data[(data["Timestamp"].isin(RBDownInterval)) & (data["Raspberry"].isin(["Raspberry B"]))] = np.nan
    data[(data["Timestamp"].isin(RCDownInterval)) & (data["Raspberry"].isin(["Raspberry C"]))] = np.nan
    data[(data["Timestamp"].isin(RDDownInterval)) & (data["Raspberry"].isin(["Raspberry D"]))] = np.nan
    data[(data["Timestamp"].isin(REDownInterval)) & (data["Raspberry"].isin(["Raspberry E"]))] = np.nan

    data.set_index("Timestamp", inplace=True)
    data.dropna(inplace=True)
    data.drop(columns=["Nº Mensajes", "Raspberry"], inplace=True)

    timestampSplit = [data.loc[date].values for date in data.index.unique()]
    #for i in range(1, len(timestampSplit)):



    #print(list[0])
    '''
    group.reset_index(inplace=True)
    groupToCheck.reset_index(inplace=True)
    count = 0
    uniqueMAC = group["MAC"].unique()
    for mac in uniqueMAC:
        if len(groupToCheck.loc[groupToCheck["MAC"] == mac]) != 0:
            count = count + 1
    totalMACPreviousInterval = count

    return totalMACPreviousInterval
    '''

def getTotalDevicesInTwoPreviousIntervals(data, timeSeries):
    """Función que devuelve una lista con el número de dispositivos registrados en el intervalo de tiempo actual y los
    dos anteriores."""

    if len(timeSeries) < 3:
        totalMACTwoPreviousInterval = 0
    else:
        group = data.loc[data["Timestamp"] == timeSeries[-1]]
        groupToCheck = data.loc[data["Timestamp"] == timeSeries[-2]]
        groupToCheckPrevious = data.loc[data["Timestamp"] == timeSeries[-3]]
        group.reset_index(inplace=True)
        groupToCheck.reset_index(inplace=True)
        groupToCheckPrevious.reset_index(inplace=True)
        uniqueMAC = group["MAC"].unique()
        count = 0
        for mac in uniqueMAC:
            if len(groupToCheck.loc[groupToCheck["MAC"] == mac]) != 0 and \
                len(groupToCheckPrevious.loc[groupToCheckPrevious["MAC"] == mac]) != 0:
                count = count + 1
        totalMACTwoPreviousInterval = count

    return totalMACTwoPreviousInterval


def getTrainingDataset(dataArray, personCountArray, stateArray):
    """Función que devuelve un conjunto de datos para el algoritmo de Machine Learning y un dataframe con los valores
    acumulados hasta ese momento"""

    columns = ["Timestamp", "Person Count", "Minutes", "N MAC TOTAL", "N MAC RA", "N MAC RB", "N MAC RC", "N MAC RD",
               "N MAC RE",
               "N MAC RDE", "N MAC RCE", "N MAC RCDE", "N MAC RBE", "N MAC MEN RA 10", "N MAC MEN RA 10-30",
               "N MAC MEN RA 30", "N MAC MEN RB 10", "N MAC MEN RB 10-30", "N MAC MEN RB 30", "N MAC MEN RC 10",
               "N MAC MEN RC 10-30", "N MAC MEN RC 30", "N MAC MEN RD 10", "N MAC MEN RD 10-30", "N MAC MEN RD 30",
               "N MAC MEN RE 10", "N MAC MEN RE 10-30", "N MAC MEN RE 30", "N MAC INTERVALO ANTERIOR",
               "N MAC DOS INTERVALOS ANTERIORES"]

    for i in range(len(dataArray)):
        data = dataArray[i]
        personCount = personCountArray[i]
        state = stateArray[i]

        RADownInterval = state.loc[state["RA(1/0)"] == 0].index
        RBDownInterval = state.loc[state["RB(1/0)"] == 0].index
        RCDownInterval = state.loc[state["RC(1/0)"] == 0].index
        RDDownInterval = state.loc[state["RD(1/0)"] == 0].index
        REDownInterval = state.loc[state["RE(1/0)"] == 0].index
        RDownInterval = (RADownInterval, RBDownInterval, RCDownInterval, RDDownInterval, REDownInterval)

        dataGroup = data.groupby("Timestamp").nunique()
        totalMAC = dataGroup["MAC"].values

        totalMACRA, totalMACRB, totalMACRC, totalMACRD, totalMACRE = getTotalDevicesByRaspberry(data, RDownInterval)

        totalMACRCDE, totalMACRCE, totalMACRDE, totalMACRBE = getTotalDevicesByPairRaspberries(data, RDownInterval)

        totalMACRA_10, totalMACRA_1030, totalMACRA_30, totalMACRB_10, totalMACRB_1030, totalMACRB_30, totalMACRC_10, \
        totalMACRC_1030, totalMACRC_30, totalMACRD_10, totalMACRD_1030, totalMACRD_30, totalMACRE_10, totalMACRE_1030, \
        totalMACRE_30 = getTotalDeviceByMessageNumber(data, RDownInterval)

        totalMACPreviousInterval = getTotalDevicesInPreviousInterval(data, RDownInterval)

    '''


    

    totalMACTwoPreviousInterval = getTotalDevicesInTwoPreviousIntervals(data, timeSeries)

    timestamp = dataNow["Timestamp"].dt.strftime('%Y-%m-%d %H:%M:%S')
    data = [timestamp.values[0], personCount["personCount"][0], personCount["Minutes"].values[0], totalMAC, totalMACRA,
            totalMACRB,
            totalMACRC, totalMACRD, totalMACRE, totalMACRDE, totalMACRCE, totalMACRCDE, totalMACRBE, totalMACRA_10,
            totalMACRA_1030, totalMACRA_30, totalMACRB_10, totalMACRB_1030, totalMACRB_30, totalMACRC_10,
            totalMACRC_1030, totalMACRC_30, totalMACRD_10, totalMACRD_1030, totalMACRD_30, totalMACRE_10,
            totalMACRE_1030, totalMACRE_30, totalMACPreviousInterval, totalMACTwoPreviousInterval]

    trainingSet = pd.DataFrame([data], columns=columns)
    trainingDataSet = pd.concat([trainingDataSet, trainingSet], ignore_index=True)

    return trainingSet, trainingDataSet
'''


def getTrainingSetFormat(trainingSet, finalTrainingDataSet, columns=None):
    """Función que devuelve un Dataframe con las columnas incluidas en el argumento columns que serán las que reciba el
    estimador finalmente. También devuelve el histórico de los anteriores resultados."""

    if columns is None:
        columns = ["N MAC RA", "N MAC RB", "N MAC RC", "N MAC RD", "N MAC RE", "N MAC RDE", "N MAC RCE", "N MAC RCDE",
                   "N MAC RBE", "N MAC MEN RA 10", "N MAC MEN RB 10", "N MAC MEN RC 10", "N MAC MEN RD 10",
                   "N MAC MEN RE 10", "N MAC INTERVALO ANTERIOR", "N MAC DOS INTERVALOS ANTERIORES"]

    finalTrainingSet = pd.DataFrame(trainingSet[["Timestamp", "Person Count", "Minutes"]],
                                    columns=["Timestamp", "Person Count", "Minutes"])
    finalTrainingSet = pd.concat([finalTrainingSet, trainingSet[columns]], axis=1)

    finalTrainingSet["Timestamp"] = pd.to_datetime(finalTrainingSet["Timestamp"])

    finalTrainingDataSet = pd.concat([finalTrainingDataSet, finalTrainingSet])

    return finalTrainingSet, finalTrainingDataSet

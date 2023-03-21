import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter

warnings.filterwarnings("ignore")


def setDateTimeLimits(data, values, day, sampling, isDf=True):
    """Función que devuelve un conjunto de datos con los límites de tiempo establecidos."""

    # Fecha y hora de inicio
    initDate = pd.to_datetime(day + " 07:00:00")
    endDate = pd.to_datetime(day + " 21:" + str(60 - sampling) + ":00")

    # Si el conjunto de datos es un DataFrame, se trata de manera distinta una Serie.
    # Se crea un Dataframe con los valores pasados por argumento junto al Timestamp de inicio y fin y se concatenan.
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
        data.set_index("Timestamp", inplace=True)

    # En caso de una Serie, se crea una Serie con los valores pasados por argumento junto al Timestamp de inicio y fin
    # y se concatenan.
    else:
        dfInit = pd.Series(values, index=[initDate], name="Timestamp")
        dfEnd = pd.Series(values, index=[endDate], name="Timestamp")
        if initDate not in data:
            data = pd.concat([dfInit, data])

        if endDate not in data:
            data = pd.concat([data, dfEnd])

    return data


def readAndPrepareDataFromDirectory(dataPath, personCountPath, sampling):
    """Función que lee los archivos de datos de los receptores Bluetooth, del contador de personas y los estados de cada
    Raspberry y los concentra en un array."""

    # Todos los archivos csv se cargarán dentro de una lista para ser utilizados posteriormente.
    dataArray = []
    personCountArray = []
    stateArray = []
    dataPath = Path(dataPath)
    personCountPath = Path(personCountPath)

    print("Cargando datos BLE...")
    # Para los datos BLE, se cargan las columnas necesarias y eliminamos MAC de señalización.
    for file in dataPath.iterdir():
        data = pd.read_csv(file, sep=";", usecols=["Timestamp int.", "Raspberry", "Nº Mensajes", "MAC"])
        data["Timestamp int."] = pd.to_datetime(data["Timestamp int."], dayfirst=True)
        data = data.rename(columns={"Timestamp int.": "Timestamp"})
        stateArray.append(getRaspberryState(data, sampling))
        data = data.drop(data[data["MAC"] == "00:00:00:00:00:00"].index).reset_index(drop=True)
        data.set_index("Timestamp", inplace=True)
        dataArray.append(data)

    print("Cargando datos del contador de personas...")
    # Para los datos del contador de personas, generando la columna Timestamp y agrupando los valores en intervalos de 5
    # minutos. En esta carga en el caso de generar un valor nulo, se interpola en caso de ser hora de estudio y posteiriormente,
    # se rellena con valores nulos para tener el mismo número de intervalos temporales que los datos BLE.
    for file in personCountPath.iterdir():
        personCount = pd.read_csv(file, sep=";")
        personCount["Timestamp"] = personCount["Fecha"] + " " + personCount["Hora"]
        personCount.drop(columns=["Fecha", "Hora"], inplace=True)
        personCount["Timestamp"] = pd.to_datetime(personCount["Timestamp"], dayfirst=True)

        # Si existe la columna Sensor, debe tratarse de manera distinta.
        if "Sensor" in personCount.columns:
            personCount = personCount.drop(personCount[personCount["Sensor"] == "KeepAlive"].index)
            personCount.replace({"Right2": "Right"}, inplace=True)
            personCount.drop_duplicates(subset=["Timestamp", "Sensor"], keep="first", inplace=True)
        personCount["Ocupacion"] = (2 * personCount["Evento In-Out(1/0)"].astype(int) - 1).cumsum()

        # Se rellena e interpola los intervalos dentro de la ventana de estudio.
        personCount = personCount.groupby(pd.Grouper(key="Timestamp", freq="5T")).last().fillna(method="ffill")
        personCount = personCount.round()

        # Se rellena los intervalos fuera de la ventana de estudio con valores nulos.
        day = personCount.index.date[0].strftime(format="%Y-%m-%d")
        personCount = personCount.loc[:, "Ocupacion"]
        personCount = setDateTimeLimits(personCount, np.nan, day, False)
        personCount = personCount.resample(str(sampling) + "T").asfreq()

        personCountArray.append(personCount)

    return dataArray, personCountArray, stateArray


def getRaspberryState(data, sampling):
    """Función que devuelve un DataFrame con los estados de cada Raspberry."""

    stateColumns = ["Timestamp", "RA(1/0)", "RB(1/0)", "RC(1/0)", "RD(1/0)", "RE(1/0)"]
    stateIds = ["RA(1/0)", "RB(1/0)", "RC(1/0)", "RD(1/0)", "RE(1/0)"]
    ids = ["Raspberry A", "Raspberry B", "Raspberry C", "Raspberry D", "Raspberry E"]

    flagGroup = data.loc[data["MAC"] == "00:00:00:00:00:00"]

    day = data["Timestamp"].iloc[0].date().strftime("%Y-%m-%d")
    dateList = pd.date_range(start=day + " 07:00:00", end=day + " 21:" + str(60 - sampling) + ":00",
                             freq=str(sampling) + "T")

    state = pd.DataFrame(np.zeros([len(dateList), len(stateColumns)]), columns=stateColumns)
    state["Timestamp"] = dateList

    for date in dateList:
        group = flagGroup.loc[flagGroup["Timestamp"] == date]
        for i in range(len(ids)):
            if group.loc[group["Raspberry"] == ids[i]]["Nº Mensajes"].values[0] != 0:
                state.loc[state["Timestamp"] == date, stateIds[i]] = 1

    return state


def parseDataByRaspberry(data):
    """Función que devuelve un conjunto de datos filtrado por cada Raspberry. Devuelve un conjunto por Raspberry."""

    dataCopy = data.copy()
    dataRA = dataCopy.loc[dataCopy["Raspberry"] == "Raspberry A"]
    dataRB = dataCopy.loc[dataCopy["Raspberry"] == "Raspberry B"]
    dataRC = dataCopy.loc[dataCopy["Raspberry"] == "Raspberry C"]
    dataRD = dataCopy.loc[dataCopy["Raspberry"] == "Raspberry D"]
    dataRE = dataCopy.loc[dataCopy["Raspberry"] == "Raspberry E"]

    return dataRA, dataRB, dataRC, dataRD, dataRE


def groupDataByRaspberryTime(data):
    """Función que devuelve conjuntos de datos con valores únicos filtrados por Raspberry y agrupados por Timestamp."""

    dataRA, dataRB, dataRC, dataRD, dataRE = parseDataByRaspberry(data)

    dataRA = dataRA.groupby("Timestamp").nunique()
    dataRB = dataRB.groupby("Timestamp").nunique()
    dataRC = dataRC.groupby("Timestamp").nunique()
    dataRD = dataRD.groupby("Timestamp").nunique()
    dataRE = dataRE.groupby("Timestamp").nunique()

    return dataRA, dataRB, dataRC, dataRD, dataRE


def getTotalDevicesByRaspberry(data, state, sampling):
    """Función que devuelve conjuntos de datos con el número de dispositivos únicos filtrados por Raspberry y agrupados
    por Timestamp."""

    # Se obtienen los datos filtrados por Raspberry y el estado de las Raspberry.
    dataRA, dataRB, dataRC, dataRD, dataRE = groupDataByRaspberryTime(data)
    RADownInterval, RBDownInterval, RCDownInterval, RDDownInterval, REDownInterval = state

    # Se eliminan las columnas innecesarias.
    dataRA.drop(columns=["Nº Mensajes", "Raspberry"], inplace=True)
    dataRB.drop(columns=["Nº Mensajes", "Raspberry"], inplace=True)
    dataRC.drop(columns=["Nº Mensajes", "Raspberry"], inplace=True)
    dataRD.drop(columns=["Nº Mensajes", "Raspberry"], inplace=True)
    dataRE.drop(columns=["Nº Mensajes", "Raspberry"], inplace=True)

    # Se agrupan en listas para poder iterar y así evitar repetir código.
    dataArray = [dataRA, dataRB, dataRC, dataRD, dataRE]
    statusList = [RADownInterval, RBDownInterval, RCDownInterval, RDDownInterval, REDownInterval]
    finalDataList = []

    day = data.index.date[0].strftime(format="%Y-%m-%d")

    # Bucle que recorre las listas anteriores y genera un conjunto de datos con el número de dispositivos únicos por
    # Raspberry y agrupados por Timestamp.
    for i, column in enumerate(dataArray):
        column = setDateTimeLimits(column, [0], day, sampling)
        column = column.resample(str(sampling) + "T").asfreq().fillna(0)
        column.loc[statusList[i], "MAC"] = np.nan

        finalDataList.append(column)

    # Destructuring de la lista para devolver los conjuntos de datos.
    totalMACRA, totalMACRB, totalMACRC, totalMACRD, totalMACRE = finalDataList

    # Finalmente, se obtiene una lista con los valores para cada Raspberry.
    totalMACRA = totalMACRA["MAC"].values
    totalMACRB = totalMACRB["MAC"].values
    totalMACRC = totalMACRC["MAC"].values
    totalMACRD = totalMACRD["MAC"].values
    totalMACRE = totalMACRE["MAC"].values

    return totalMACRA, totalMACRB, totalMACRC, totalMACRD, totalMACRE


def getTotalDevicesByPairRaspberry(data, state, sampling):
    """Función que devuelve cuatro listas compuestas por los dispositivos captados en el mismo intervalo de tiempo por
    las parejas C-E, D-E, B-E y el trío C-D-E."""

    # Se obtienen los datos filtrados por Raspberry y el estado de las Raspberry.
    dataRA, dataRB, dataRC, dataRD, dataRE = parseDataByRaspberry(data)
    RADownInterval, RBDownInterval, RCDownInterval, RDDownInterval, REDownInterval = state

    # Se elimina la columna innecesaria.
    nDevicesIntervalDataRAMerge = dataRA.drop(columns="Nº Mensajes")
    nDevicesIntervalDataRBMerge = dataRB.drop(columns="Nº Mensajes")
    nDevicesIntervalDataRCMerge = dataRC.drop(columns="Nº Mensajes")
    nDevicesIntervalDataRDMerge = dataRD.drop(columns="Nº Mensajes")
    nDevicesIntervalDataREMerge = dataRE.drop(columns="Nº Mensajes")

    # Merge de los datos de todas las Raspberry.
    nDevicesIntervalDataRDEMerge = nDevicesIntervalDataRDMerge.merge(nDevicesIntervalDataREMerge, how="outer",
                                                                     on=("Timestamp", "MAC"), copy=False,
                                                                     suffixes=("_d", "_e"))
    nDevicesIntervalDataRCDEMerge = nDevicesIntervalDataRDEMerge.merge(nDevicesIntervalDataRCMerge, how="outer",
                                                                       on=("Timestamp", "MAC"), copy=False)
    nDevicesIntervalDataRBCDEMerge = nDevicesIntervalDataRCDEMerge.merge(nDevicesIntervalDataRBMerge, how="outer",
                                                                         on=("Timestamp", "MAC"), copy=False,
                                                                         suffixes=("_c", "_b"))
    nDevicesIntervalDataRABCDEMerge = nDevicesIntervalDataRBCDEMerge.merge(nDevicesIntervalDataRAMerge, how="outer",
                                                                           on=("Timestamp", "MAC"), copy=False)

    # Los datos se agrupan por Timestamp y MAC.
    group = nDevicesIntervalDataRABCDEMerge.groupby(["Timestamp", "MAC"]).nunique()

    # Cada grupo está generado en función de que se cumplan las condiciones de que el dispositivo esté presente en
    # las Raspberry que forman la pareja o trío.
    group_CDE = group.loc[(group["Raspberry_c"] == 1) & (group["Raspberry_d"] == 1) & (group["Raspberry_e"] == 1)]
    group_CE = group.loc[(group["Raspberry_c"] == 1) & (group["Raspberry_e"] == 1)]
    group_DE = group.loc[(group["Raspberry_d"] == 1) & (group["Raspberry_e"] == 1)]
    group_BE = group.loc[(group["Raspberry_b"] == 1) & (group["Raspberry_e"] == 1)]

    day = group.index.get_level_values(0).date[0].strftime(format="%Y-%m-%d")

    # Se agrupan en una lista para poder iterar y así evitar repetir código.
    dataArray = [group_CDE, group_CE, group_DE, group_BE]
    finalDataList = []

    # Al resetear el índice, la columna Timestamp repite sus valores tantas veces como direcciones MAC haya en ese intervalo.
    # Contando el número de veces que se repite cada valor de Timestamp, se obtiene el número de dispositivos únicos
    # en cada intervalo de tiempo.
    for i, column in enumerate(dataArray):
        column.reset_index(inplace=True)
        column = column["Timestamp"].value_counts(sort=False)
        column = setDateTimeLimits(column, 0, day, False)
        column = column.resample(str(sampling) + "T").asfreq().fillna(0)

        # En función de que conjunto se esté procesando, aplican unos estados u otros.
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

    # Destructuring de la lista para devolver los conjuntos de datos.
    totalMACRCDE, totalMACRCE, totalMACRDE, totalMACRBE = finalDataList

    return totalMACRCDE.values, totalMACRCE.values, totalMACRDE.values, totalMACRBE.values


def getDevicesByMessageRange(dataArray):
    """Función que devuelve una lista con el número de dispositivos captados en función del número de mensajes recibidos."""

    # Busca los dispositivos que cumplen las condiciones de que el número de mensajes recibidos esté en el rango
    # especificado.
    finalDataList = []
    for data in dataArray:
        totalMACR_10 = data.loc[data["Nº Mensajes"] <= 10]
        finalDataList.append(totalMACR_10)
        totalMACR_1030 = data.loc[(data["Nº Mensajes"] > 10) & (data["Nº Mensajes"] <= 30)]
        finalDataList.append(totalMACR_1030)
        totalMACR_30 = data.loc[data["Nº Mensajes"] > 30]
        finalDataList.append(totalMACR_30)

    return finalDataList


def getTotalDeviceByMessageNumber(data, state, sampling):
    """Función que devuelve tres listas por Raspberry, una por intervalo de número de mensajes por debajo
    de 10, entre 10 y 30 y superior a 30."""

    # Se obtienen los datos filtrados por Raspberry y el estado de las Raspberry.
    dataRA, dataRB, dataRC, dataRD, dataRE = parseDataByRaspberry(data)
    RADownInterval, RBDownInterval, RCDownInterval, RDDownInterval, REDownInterval = state

    # Los datos se agrupan por Timestamp y MAC con la suma acumulada de mensajes.
    dataRA = dataRA.groupby(["Timestamp", "MAC"]).sum()
    dataRB = dataRB.groupby(["Timestamp", "MAC"]).sum()
    dataRC = dataRC.groupby(["Timestamp", "MAC"]).sum()
    dataRD = dataRD.groupby(["Timestamp", "MAC"]).sum()
    dataRE = dataRE.groupby(["Timestamp", "MAC"]).sum()

    # Se obtienen los datos filtrados por número de mensajes recibidos.
    dataArray = getDevicesByMessageRange([dataRA, dataRB, dataRC, dataRD, dataRE])

    day = data.index.date[0].strftime(format="%Y-%m-%d")

    finalDataList = []

    # Al resetear el índice, la columna Timestamp repite sus valores tantas veces como direcciones MAC haya en ese intervalo.
    # Contando el número de veces que se repite cada valor de Timestamp, se obtiene el número de dispositivos únicos
    # en cada intervalo de tiempo.
    for i, column in enumerate(dataArray):
        column.reset_index(inplace=True)
        column = column["Timestamp"].value_counts(sort=False)
        column = setDateTimeLimits(column, 0, day, False)
        column = column.resample(str(sampling) + "T").asfreq().fillna(0)

        # En función de que conjunto se esté procesando, aplican unos estados u otros.
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

    # Destructuring de la lista para devolver los conjuntos de datos.
    totalMACRA_10, totalMACRA_1030, totalMACRA_30, totalMACRB_10, totalMACRB_1030, totalMACRB_30, totalMACRC_10, \
        totalMACRC_1030, totalMACRC_30, totalMACRD_10, totalMACRD_1030, totalMACRD_30, totalMACRE_10, \
        totalMACRE_1030, totalMACRE_30 = finalDataList

    return totalMACRA_10, totalMACRA_1030, totalMACRA_30, totalMACRB_10, totalMACRB_1030, totalMACRB_30, totalMACRC_10, \
        totalMACRC_1030, totalMACRC_30, totalMACRD_10, totalMACRD_1030, totalMACRD_30, totalMACRE_10, \
        totalMACRE_1030, totalMACRE_30


def getTotalDevicesInPreviousInterval(data, state, sampling):
    """Función que devuelve una lista con el número de dispositivos registrados en el intervalo de tiempo actual y el
    anterior."""

    dataCopy = data.copy()

    # Se obtiene el estado de las Raspberry.
    RADownInterval, RBDownInterval, RCDownInterval, RDDownInterval, REDownInterval = state

    invalidDates = set(RADownInterval) & set(RBDownInterval) & set(RCDownInterval) & set(RDDownInterval) & set(
        REDownInterval)

    # Búsqueda en función del intervalo de tiempo y de Raspberry para comprobar su estado.
    dataCopy[(dataCopy.index.isin(RADownInterval)) & (dataCopy["Raspberry"].isin(["Raspberry A"]))] = np.nan
    dataCopy[(dataCopy.index.isin(RBDownInterval)) & (dataCopy["Raspberry"].isin(["Raspberry B"]))] = np.nan
    dataCopy[(dataCopy.index.isin(RCDownInterval)) & (dataCopy["Raspberry"].isin(["Raspberry C"]))] = np.nan
    dataCopy[(dataCopy.index.isin(RDDownInterval)) & (dataCopy["Raspberry"].isin(["Raspberry D"]))] = np.nan
    dataCopy[(dataCopy.index.isin(REDownInterval)) & (dataCopy["Raspberry"].isin(["Raspberry E"]))] = np.nan

    day = dataCopy.index.date[0].strftime("%Y-%m-%d")

    # Se eliminan los valores nulos y las columnas innecesarias.
    dataCopy.dropna(inplace=True)
    dataCopy.drop(columns=["Nº Mensajes", "Raspberry"], inplace=True)

    timestampSplit = []

    # Bucle que crea una lista por cada intervalo de tiempo, agrupándolos en una lista.
    for date in dataCopy.index.unique():
        if len(dataCopy.loc[date]) == 1:
            timestampSplit.append(np.array([dataCopy.loc[date]["MAC"]], dtype=object))
        else:
            timestampSplit.append(dataCopy.loc[date]["MAC"].unique())

    # En el primer instante, no es posible comparar con el anterior, por lo que se añade un valor 0.
    totalMACPreviousInterval = pd.DataFrame([[pd.to_datetime(day + " 07:00:00"), 0]], columns=["Timestamp", "MAC"])

    # Bucle que recorre la lista de intervalos de tiempo y compara las MAC de cada uno con los del anterior comprobando
    # coincidentes.
    for i in range(1, len(timestampSplit)):
        date = dataCopy.index.unique()[i]
        coincidences = len(set(timestampSplit[i]) & set(timestampSplit[i - 1]))
        actualDf = pd.DataFrame([[date, coincidences]], columns=["Timestamp", "MAC"])
        totalMACPreviousInterval = pd.concat([totalMACPreviousInterval, actualDf])

    # Se establece las fechas límite y se rellenan los huecos.
    totalMACPreviousInterval.set_index("Timestamp", inplace=True)
    totalMACPreviousInterval = setDateTimeLimits(totalMACPreviousInterval, [0], day, sampling)
    totalMACPreviousInterval = totalMACPreviousInterval.resample(str(sampling) + "T").asfreq().fillna(0)
    totalMACPreviousInterval.loc[invalidDates] = np.nan
    totalMACPreviousInterval = np.array(totalMACPreviousInterval["MAC"].values)

    return totalMACPreviousInterval


def getTotalDevicesInTwoPreviousIntervals(data, state, sampling):
    """Función que devuelve una lista con el número de dispositivos registrados en el intervalo de tiempo actual y los
    dos anteriores."""

    dataCopy = data.copy()

    # Se obtiene el estado de las Raspberry.
    RADownInterval, RBDownInterval, RCDownInterval, RDDownInterval, REDownInterval = state

    invalidDates = set(RADownInterval) & set(RBDownInterval) & set(RCDownInterval) & set(RDDownInterval) & set(
        REDownInterval)

    # Búsqueda en función del intervalo de tiempo y de Raspberry para comprobar su estado.
    dataCopy[(dataCopy.index.isin(RADownInterval)) & (dataCopy["Raspberry"].isin(["Raspberry A"]))] = np.nan
    dataCopy[(dataCopy.index.isin(RBDownInterval)) & (dataCopy["Raspberry"].isin(["Raspberry B"]))] = np.nan
    dataCopy[(dataCopy.index.isin(RCDownInterval)) & (dataCopy["Raspberry"].isin(["Raspberry C"]))] = np.nan
    dataCopy[(dataCopy.index.isin(RDDownInterval)) & (dataCopy["Raspberry"].isin(["Raspberry D"]))] = np.nan
    dataCopy[(dataCopy.index.isin(REDownInterval)) & (dataCopy["Raspberry"].isin(["Raspberry E"]))] = np.nan

    day = dataCopy.index.date[0].strftime("%Y-%m-%d")

    # Se eliminan los valores nulos y las columnas innecesarias.
    dataCopy.dropna(inplace=True)
    dataCopy.drop(columns=["Nº Mensajes", "Raspberry"], inplace=True)

    # Bucle que crea una lista por cada intervalo de tiempo, agrupándolos en una lista.
    timestampSplit = []
    for date in dataCopy.index.unique():
        if len(dataCopy.loc[date]) == 1:
            timestampSplit.append(np.array([dataCopy.loc[date]["MAC"]], dtype=object))
        else:
            timestampSplit.append(dataCopy.loc[date]["MAC"].unique())

    # En el primer y segundo instante, no es posible comparar con los anteriores, por lo que se añade un valor 0 en ambos.
    totalMACTwoPreviousInterval = pd.DataFrame(
        [[pd.to_datetime(day + " 07:00:00"), 0], [pd.to_datetime(day + " 07:0" + str(sampling) + ":00"), 0]],
        columns=["Timestamp", "MAC"])

    # Bucle que recorre la lista de intervalos de tiempo y compara las MAC de cada uno con los de los dos anteriores comprobando
    # coincidentes.
    for i in range(2, len(timestampSplit)):
        date = dataCopy.index.unique()[i]
        coincidences = len(set(timestampSplit[i]) & set(timestampSplit[i - 1]) & set(timestampSplit[i - 2]))
        actualDf = pd.DataFrame([[date, coincidences]], columns=["Timestamp", "MAC"])
        totalMACTwoPreviousInterval = pd.concat([totalMACTwoPreviousInterval, actualDf])

    # Se establece las fechas límite y se rellenan los huecos.
    totalMACTwoPreviousInterval.set_index("Timestamp", inplace=True)
    totalMACTwoPreviousInterval = setDateTimeLimits(totalMACTwoPreviousInterval, [0], day, sampling)
    totalMACTwoPreviousInterval = totalMACTwoPreviousInterval.resample(str(sampling) + "T").asfreq().fillna(0)
    totalMACTwoPreviousInterval.loc[invalidDates] = np.nan
    totalMACTwoPreviousInterval = np.array(totalMACTwoPreviousInterval["MAC"].values)

    return totalMACTwoPreviousInterval


def savePlotColumns(data, path, categoryName):
    """Función que guarda en una carpeta las gráficas para cada una de las columnas del training set."""

    day = data["Timestamp"].iloc[0].date().strftime("%Y-%m-%d")
    imgFolder = Path("../figures", path, categoryName, day)
    if not imgFolder.exists():
        imgFolder.mkdir(parents=True)

    # Para cada columna del Dataframe pasado como argumento, se grafica en comparación de la ocupación en función del tiempo.
    for i in range(3, len(data.columns)):
        name = data.columns[i]
        imgName = data.columns[i] + ".jpg"

        fig, ax = plt.subplots(figsize=(10, 6))
        date_form = DateFormatter("%H:%M")
        ax.xaxis.set_major_formatter(date_form)
        plt.plot(data["Timestamp"], data["Ocupacion"], label="Ocupacion", color="red")
        plt.plot(data["Timestamp"], data[data.columns[i]], label=data.columns[i], color="blue")
        plt.xlabel("Timestamp")
        plt.ylabel("Devices")
        plt.legend()
        plt.title(name)
        plt.grid()
        plt.savefig(Path(imgFolder, imgName))
        plt.clf()
        plt.close()


def fillSet(data, dates, sampling):
    """Función que completa el conjunto de datos filtrado rellenando los valores nulos o eliminando las filas
    imposibles de interpolar."""

    dataCopy = data.copy()

    # Se interpolan los valores en los que sea posible.
    dataCopy.set_index("Timestamp", inplace=True)
    filledSet = dataCopy.resample(str(sampling) + "T").mean().interpolate()

    # En las fechas donde hubo fallos, se eliminan las filas.
    filledSet.loc[dates] = np.nan
    filledSet.reset_index(inplace=True)
    filledSet = filledSet.round(0)
    filledSet.dropna(inplace=True)
    return filledSet


def getDataset(dataArray, personCountArray, stateArray, categoryName, path2, path3, sampling, path1="../results"):
    """Función que devuelve un conjunto de datos para el algoritmo de Machine Learning y un dataframe con los valores
    acumulados hasta ese momento."""

    path1 = Path(path1)

    if not path1.exists():
        path1.mkdir(parents=True)

    # Columnas calculadas a partir de los datos de entrada.
    columns = ["Timestamp", "Ocupacion", "Minutes", "N MAC TOTAL", "N MAC RA", "N MAC RB", "N MAC RC", "N MAC RD",
               "N MAC RE",
               "N MAC RDE", "N MAC RCE", "N MAC RCDE", "N MAC RBE", "N MAC MEN RA 10", "N MAC MEN RA 10-30",
               "N MAC MEN RA 30", "N MAC MEN RB 10", "N MAC MEN RB 10-30", "N MAC MEN RB 30", "N MAC MEN RC 10",
               "N MAC MEN RC 10-30", "N MAC MEN RC 30", "N MAC MEN RD 10", "N MAC MEN RD 10-30", "N MAC MEN RD 30",
               "N MAC MEN RE 10", "N MAC MEN RE 10-30", "N MAC MEN RE 30", "N MAC INTERVALO ANTERIOR",
               "N MAC DOS INTERVALOS ANTERIORES"]

    # Columnas del conjunto de aprendizaje
    columnsFilter = ["Timestamp", "Ocupacion", "Minutes", "N MAC RA", "N MAC RB", "N MAC RC", "N MAC RD", "N MAC RE",
                     "N MAC RDE", "N MAC RCE", "N MAC RCDE",
                     "N MAC RBE", "N MAC MEN RA 10", "N MAC MEN RB 10", "N MAC MEN RC 10", "N MAC MEN RD 10",
                     "N MAC MEN RE 10", "N MAC INTERVALO ANTERIOR", "N MAC DOS INTERVALOS ANTERIORES"]

    rawDataSet = pd.DataFrame(columns=columns)
    filterDataSet = pd.DataFrame(columns=columnsFilter)
    filledDataSet = pd.DataFrame(columns=columnsFilter)

    # En intervalos de cinco minutos, se crea la columna Minutes que indica el número de minutos transcurridos desde
    # las 7:00.
    length = len(personCountArray[0].index.unique())
    minutes = np.linspace(0, (length - 1) * sampling, length, dtype=int)

    # Para cada uno de los archivos cargados, es decir, días, se crea un dataframe con las columnas calculadas.
    for i in range(len(dataArray)):
        data = dataArray[i]
        personCount = personCountArray[i]
        state = stateArray[i]

        day = data.index[0].date().strftime("%Y-%m-%d")

        # Se cargan los estados de cada Raspberry y se agrupan en un tuple.
        RADownInterval = state.loc[state["RA(1/0)"] == 0].index
        RBDownInterval = state.loc[state["RB(1/0)"] == 0].index
        RCDownInterval = state.loc[state["RC(1/0)"] == 0].index
        RDDownInterval = state.loc[state["RD(1/0)"] == 0].index
        REDownInterval = state.loc[state["RE(1/0)"] == 0].index
        RDownInterval = (RADownInterval, RBDownInterval, RCDownInterval, RDDownInterval, REDownInterval)

        invalidDates = set(RADownInterval) & set(RBDownInterval) & set(RCDownInterval) & set(RDDownInterval) & set(
            REDownInterval)

        # Añadimos valores nulos donde el estado de la Raspberry es 0.
        dataGroup = data.copy()
        dataGroup[
            (dataGroup.index.isin(RADownInterval)) & (dataGroup["Raspberry"].isin(["Raspberry A"]))] = np.nan
        dataGroup[
            (dataGroup.index.isin(RBDownInterval)) & (dataGroup["Raspberry"].isin(["Raspberry B"]))] = np.nan
        dataGroup[
            (dataGroup.index.isin(RCDownInterval)) & (dataGroup["Raspberry"].isin(["Raspberry C"]))] = np.nan
        dataGroup[
            (dataGroup.index.isin(RDDownInterval)) & (dataGroup["Raspberry"].isin(["Raspberry D"]))] = np.nan
        dataGroup[
            (dataGroup.index.isin(REDownInterval)) & (dataGroup["Raspberry"].isin(["Raspberry E"]))] = np.nan

        # Se agrupa en función del Timestamp y se calculan los dispositivos únicos en cada intervalo.
        dataGroup = dataGroup.groupby("Timestamp").nunique()
        dataGroup = setDateTimeLimits(dataGroup, [np.nan, np.nan, np.nan], day, sampling)
        dataGroup = dataGroup.resample(str(sampling) + "T").asfreq()
        totalMAC = dataGroup["MAC"].values

        # Se calculan los dispositivos únicos en cada Raspberry.
        totalMACRA, totalMACRB, totalMACRC, totalMACRD, totalMACRE = getTotalDevicesByRaspberry(data, RDownInterval,
                                                                                                sampling)

        # Se calculan los dispositivos únicos por par y trio de Raspberry.
        totalMACRCDE, totalMACRCE, totalMACRDE, totalMACRBE = getTotalDevicesByPairRaspberry(data, RDownInterval,
                                                                                             sampling)

        # Se calculan los dispositivos únicos en función del número de mensajes por Raspberry.
        totalMACRA_10, totalMACRA_1030, totalMACRA_30, totalMACRB_10, totalMACRB_1030, totalMACRB_30, totalMACRC_10, \
            totalMACRC_1030, totalMACRC_30, totalMACRD_10, totalMACRD_1030, totalMACRD_30, totalMACRE_10, totalMACRE_1030, \
            totalMACRE_30 = getTotalDeviceByMessageNumber(data, RDownInterval, sampling)

        # Se calcula el número de dispositivos únicos en el intervalo anterior.
        totalMACPreviousInterval = getTotalDevicesInPreviousInterval(data, RDownInterval, sampling)

        # Se calcula el número de dispositivos únicos en los dos intervalos anteriores.
        totalMACTwoPreviousInterval = getTotalDevicesInTwoPreviousIntervals(data, RDownInterval, sampling)

        # Se crea la lista de valores completos de Timestamp.
        timestamp = data.index.unique()
        timestamp = pd.Series(np.zeros(len(timestamp)), index=timestamp, name="Timestamp")
        timestamp = setDateTimeLimits(timestamp, 0, day, False)
        timestamp = timestamp.resample(str(sampling) + "T").asfreq()
        timestamp = timestamp.index.strftime("%Y-%m-%d %H:%M:%S")

        # Se crea el dataframe con las columnas calculadas.
        dataSet = np.array(np.transpose([timestamp, personCount.values, minutes, totalMAC, totalMACRA, totalMACRB,
                                         totalMACRC, totalMACRD, totalMACRE, totalMACRDE, totalMACRCE, totalMACRCDE,
                                         totalMACRBE, totalMACRA_10, totalMACRA_1030, totalMACRA_30, totalMACRB_10,
                                         totalMACRB_1030, totalMACRB_30, totalMACRC_10, totalMACRC_1030, totalMACRC_30,
                                         totalMACRD_10, totalMACRD_1030, totalMACRD_30, totalMACRE_10, totalMACRE_1030,
                                         totalMACRE_30, totalMACPreviousInterval, totalMACTwoPreviousInterval]))

        # Se concatena el Dataframe creado al Dataframe que contiene todos los datos y se grafica.
        dataSet = pd.DataFrame(dataSet, columns=columns)
        dataSet["Timestamp"] = pd.to_datetime(dataSet["Timestamp"])

        rawDataSet = pd.concat([rawDataSet, dataSet], ignore_index=True)

        print("Guardando gráficas de los datos calculados de la fecha " + day + "...")
        savePlotColumns(dataSet, path2, categoryName)

        # Se concatena el Dataframe creado al Dataframe que contiene los datos resumidos.
        filterSet = dataSet[columnsFilter]
        filterDataSet = pd.concat([filterDataSet, filterSet], ignore_index=True)

        # Se concatena el Dataframe creado al Dataframe que contiene todos los datos finales y se grafica.
        filledSet = fillSet(filterSet, invalidDates, sampling)
        filledDataSet = pd.concat([filledDataSet, filledSet], ignore_index=True)

        print("Guardando gráficas de los datos limpios de la fecha " + day + "...")
        savePlotColumns(filledSet, path3, categoryName)

    name1 = categoryName + "-set.csv"
    name2 = "filter-" + categoryName + "-set.csv"
    name3 = "filled-" + categoryName + "-set.csv"

    path1 = Path(path1, categoryName)
    if not path1.exists():
        path1.mkdir(parents=True)

    # Se guardan los datos en archivos csv.
    rawDataSet.to_csv(Path(path1, name1), sep=";", na_rep="NaN", index=False)
    filterDataSet.to_csv(Path(path1, name2), sep=";", na_rep="NaN", index=False)
    filledDataSet.to_csv(Path(path1, name3), sep=";", na_rep="NaN", index=False)

    return rawDataSet, filterDataSet, filledDataSet

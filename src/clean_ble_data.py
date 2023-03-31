import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)


def cleanBLEData(dataPath, macList, sampling, tagBle):
    """Función que limpia los datos BLE brutos y guarda los resultados en un archivo CSV."""

    # Se obtiene la lista de archivos a procesar.
    dataPath = Path(dataPath)
    macList = pd.read_csv(macList, delimiter=";", usecols=["MAC"])

    columns = ["Indice int. muestreo", "Timestamp int.", "Raspberry", "Nº Mensajes", "MAC", "Tipo MAC", "Tipo ADV",
               "BLE Size", "RSP Size", "BLE Data", "RSSI promedio"]

    ids = ["Raspberry A", "Raspberry B", "Raspberry C", "Raspberry D", "Raspberry E"]

    n = len(ids)

    # Para cada archivo disponible en la carpeta de datos.
    for file in dataPath.iterdir():
        dataBle = pd.read_csv(file, sep=";")

        # Se eliminan las direcciones MAC que se encuentran en la lista de MAC a filtrar.
        dataBle = dataBle[~dataBle["MAC"].isin(macList["MAC"])]

        # Se renombran los identificadores de las Raspberry Pi.
        dataBle["Id"] = dataBle["Id"].replace(["Raspberry1", "Raspberry2", "Raspberry3", "Raspberry5", "Raspberry7"],
                                              ["Raspberry A", "Raspberry D", "Raspberry B", "Raspberry E",
                                               "Raspberry C"])

        # Se añade columna Timestamp.
        dataBle["Timestamp int."] = pd.to_datetime(dataBle["Fecha"] + " " + dataBle["Hora"], dayfirst=True)
        day = dataBle["Timestamp int."].iloc[0].date().strftime("%Y-%m-%d")
        dataBle["Mensajes"] = 1

        # Lista con todos los Timestamps posibles, añadiendoles índice.
        dateList = enumerate(
            pd.date_range(start=day + " 07:00:00", end=day + " 21:" + str(60 - sampling) + ":00",
                          freq=str(sampling) + "T"))
        dateList = np.array([[x[0] + 1, x[1]] for x in dateList])

        # Se agrupan los valores, calculando el promedio de RSSI y añadiendo el índice de cada intervalo.
        dataBle = dataBle.groupby(
            [pd.Grouper(key="Timestamp int.", freq=str(sampling) + "T"), "Id", "MAC", "Tipo MAC", "Tipo ADV",
             "ADV Size", "RSP Size", "Advertisement"]).sum()
        dataBle["RSSI"] = np.round(dataBle["RSSI"] / dataBle["Mensajes"], 2)
        dataBle["Indice int. muestreo"] = dataBle.apply(lambda x: dateList[dateList[:, 1] == x.name[0]][0][0],
                                                        axis=1)
        dataBle.reset_index(inplace=True)
        dataBle.rename(
            columns={"Mensajes": "Nº Mensajes", "ADV Size": "BLE Size", "Advertisement": "BLE Data", "Id": "Raspberry",
                     "RSSI": "RSSI promedio"}, inplace=True)

        # Se obtiene las horas para las que existe la MAC virtual y el número de Raspberry Pi que la han enviado.
        flagGroup = dataBle.loc[dataBle["MAC"] == "00:00:00:00:00:00"]
        flagGroup = flagGroup.groupby("Timestamp int.")["MAC"].count().to_frame()
        flagGroup.reset_index(inplace=True)

        # Comprueba si los intervalos tienen tantas MAC virtuales como Raspberry Pi.
        validDates = flagGroup.loc[flagGroup["MAC"] == n]["Timestamp int."].to_list()
        datesToCheck = [x for x in dateList if x[1] not in validDates]

        timestampsList = []
        idsList = []
        intervalList = []

        # Se crean las listas de la misma longitud, una para las horas, otra para los identificadores de Raspberry Pi y
        # otra para los índices de intervalo.
        for date in datesToCheck:
            group = dataBle.loc[dataBle["Timestamp int."] == date[1]]
            groupToCheck = group.loc[dataBle["MAC"] == "00:00:00:00:00:00"]
            if groupToCheck.empty:
                idsList += ids
                timestampsList += [date[1]] * n
                intervalList += [date[0]] * n
            else:
                for rasp in ids:
                    if rasp not in groupToCheck["Raspberry"].to_list():
                        idsList.append(rasp)
                        timestampsList.append(date[1])
                        intervalList.append(date[0])

        # Se añaden los datos de las Raspberry Pi que no han enviado ningún mensaje.
        k = len(idsList)
        dataBle = pd.concat([dataBle, pd.DataFrame(
            {"Indice int. muestreo": intervalList, "Timestamp int.": timestampsList, "Raspberry": idsList,
             "Nº Mensajes": [0] * k, "MAC": ["00:00:00:00:00:00"] * k, "Tipo MAC": ["Public"] * k,
             "Tipo ADV": ["ADV_IND"] * k, "BLE Size": [4] * k, "RSP Size": [0] * k, "BLE Data": ["abcd"] * k,
             "RSSI promedio": [0] * k})], ignore_index=True)

        # Se ordena el Dataframe tanto por filas como por columnas.
        dataBle.sort_values(by=["Indice int. muestreo", "Raspberry"], inplace=True)
        dataBle = dataBle[columns]

        # Si no existe el directorio de salida, se crea.
        pathBle = Path("../results", tagBle)
        if not pathBle.exists():
            pathBle.mkdir(parents=True)

        # Se guarda el dataframe en un archivo CSV.
        pathBle = Path(pathBle, "ble-filter-clean-P_" + day + ".csv")
        dataBle.to_csv(pathBle, sep=";", index=False)

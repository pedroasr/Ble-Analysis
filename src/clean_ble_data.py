import datetime as dt
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
        dataBle["Timestamp"] = pd.to_datetime(dataBle["Fecha"] + " " + dataBle["Hora"], dayfirst=True)
        day = dataBle["Timestamp"].iloc[0].date().strftime("%Y-%m-%d")
        initDate = dataBle["Timestamp"].min()
        initDate = initDate - dt.timedelta(minutes=sampling - 1, seconds=59)
        dataBle["Mensajes"] = 1

        # Lista con todos los Timestamps posibles, añadiendoles índice.
        fullDateList = list(
            enumerate(pd.date_range(start=day + " 07:00:00", end=day + " 22:00:00", freq=str(sampling) + "T")))

        # Filtrado para obtener los Timestamps coincidentes con los datos
        dateList = np.array([x for x in fullDateList if initDate <= x[1]])
        filterData = pd.DataFrame(columns=columns)

        # Se añaden los datos de las Raspberry Pi que no han enviado ningún mensaje.
        if dateList[0][0] != 0:
            datesNotWork = fullDateList[0:dateList[0][0]]
            for date in datesNotWork:
                for rasp in ids:
                    stateData = pd.DataFrame({"Indice int. muestreo": date[0] + 1, "Timestamp int.": date[1],
                                              "Raspberry": rasp, "Nº Mensajes": 0, "MAC": "00:00:00:00:00:00",
                                              "Tipo MAC": "Public", "Tipo ADV": "ADV_IND", "BLE Size": 4, "RSP Size": 0,
                                              "BLE Data": "abcd", "RSSI promedio": 0}, index=[0])

                    filterData = pd.concat([filterData, stateData], ignore_index=True)

        # Para cada fecha disponible en la lista de fechas.
        for i, date in dateList:
            # Se obtiene la siguiente fecha a la actual. Se filtra el dataframe original entre cada pareja de fechas.
            nextDate = [x for x in dateList if x[0] == i + 1]
            try:
                group = dataBle.loc[(dataBle["Timestamp"] >= date) & (dataBle["Timestamp"] < nextDate[0][1])]
            except IndexError:
                break

            # Se agrupan los valores, calculando el promedio de RSSI.
            group = group.groupby(["Id", "MAC", "Tipo MAC", "Tipo ADV", "ADV Size", "RSP Size", "Advertisement"]).sum()
            group["RSSI promedio"] = np.round(group["RSSI"] / group["Mensajes"], 2)
            group["Timestamp"] = date.strftime("%Y-%m-%d %H:%M:%S")
            group["Indice int. muestreo"] = i + 1
            group.reset_index(inplace=True)

            # Se genera la lista de estados de las Raspberry, identificando la MAC de señalización y añadiendo el estado
            # de cada Raspberry.
            flagGroup = group.loc[group["MAC"] == "00:00:00:00:00:00"]
            activeRaspberry = list(flagGroup["Id"].unique())

            # Se añade la MAC de señalización nula a la Raspberry que no ha emitido dicha trama.

            for rasp in ids:
                if rasp not in activeRaspberry:
                    filterData = pd.concat([filterData, pd.DataFrame(
                        {"Indice int. muestreo": i + 1, "Timestamp int.": date.strftime("%Y-%m-%d %H:%M:%S"),
                         "Raspberry": rasp, "Nº Mensajes": 0, "MAC": "00:00:00:00:00:00", "Tipo MAC": "Public",
                         "Tipo ADV": "ADV_IND", "BLE Size": 4, "RSP Size": 0, "BLE Data": "abcd", "RSSI promedio": 0},
                        index=[0])], ignore_index=True)

            # Se genera la lista de datos limpios y se añade al dataframe, concatenandolos al final.
            data = np.transpose(np.array(
                [group["Indice int. muestreo"], group["Timestamp"], group["Id"], group["Mensajes"], group["MAC"],
                 group["Tipo MAC"], group["Tipo ADV"], group["ADV Size"], group["RSP Size"], group["Advertisement"],
                 group["RSSI promedio"]]))

            filtData = pd.DataFrame(data, columns=columns)
            filterData = pd.concat([filterData, filtData], ignore_index=True)

        # Si no existe el directorio de salida, se crea.
        pathBle = Path("../results", tagBle)
        if not pathBle.exists():
            pathBle.mkdir(parents=True)

        # Se guarda el dataframe en un archivo CSV.
        pathBle = Path(pathBle, "ble-filter-clean-P_" + day + ".csv")
        filterData.to_csv(pathBle, sep=";", index=False)

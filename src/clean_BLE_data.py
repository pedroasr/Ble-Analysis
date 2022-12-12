import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def cleanBLEData(dataPath, macList, sampling, tag):
    """Función que limpia los datos BLE brutos y guarda los resultados en un archivo CSV."""

    dataPath = Path(dataPath)
    macList = pd.read_csv(macList, delimiter=";", usecols=["MAC"])

    columns = ["Indice int. muestreo", "Timestamp int.", "Raspberry", "Nº Mensajes", "MAC",
               "Tipo MAC", "Tipo ADV", "BLE Size", "RSP Size", "BLE Data", "RSSI promedio"]

    for file in dataPath.iterdir():
        dataBle = pd.read_csv(file, sep=";")
        dataBle = dataBle[~dataBle.MAC.isin(macList.MAC)]

        dataBle["Id"] = dataBle["Id"].replace(["Raspberry1", "Raspberry2", "Raspberry3", "Raspberry5", "Raspberry7"],
                                              ["Raspberry A", "Raspberry D", "Raspberry B", "Raspberry E",
                                               "Raspberry C"])

        dataBle["Timestamp"] = pd.to_datetime(dataBle["Fecha"] + " " + dataBle["Hora"], dayfirst=True)
        day = dataBle["Timestamp"][0].date().strftime("%Y-%m-%d")
        initDate = dataBle["Timestamp"].min()
        endDate = dataBle["Timestamp"].max()
        dataBle["Mensajes"] = 1

        fullDateList = list(enumerate(
            pd.date_range(start=dataBle["Timestamp"].iloc[0].date().strftime("%Y-%m-%d") + " 07:00:00",
                          end=dataBle["Timestamp"].iloc[-1].date().strftime("%Y-%m-%d") + " 21:55:00",
                          freq=str(sampling) + "T")))

        dateList = np.array([x for x in fullDateList if initDate <= x[1] <= endDate])
        filterData = pd.DataFrame(columns=columns)

        for i, date in dateList:
            nextDate = [x for x in dateList if x[0] == i + 1]
            try:
                group = dataBle.loc[(dataBle["Timestamp"] >= date) & (dataBle["Timestamp"] < nextDate[0][1])]
            except IndexError:
                break

            group = group.groupby(["Id", "MAC", "Tipo MAC", "Tipo ADV", "ADV Size", "RSP Size", "Advertisement"]).sum()
            group["RSSI promedio"] = np.round(group["RSSI"] / group["Mensajes"], 2)
            group["Timestamp"] = date.strftime("%Y-%m-%d %H:%M:%S")
            group["Indice int. muestreo"] = i + 1
            group.reset_index(inplace=True)

            data = np.transpose(np.array(
                [group["Indice int. muestreo"], group["Timestamp"], group["Id"], group["Mensajes"], group["MAC"],
                 group["Tipo MAC"],
                 group["Tipo ADV"], group["ADV Size"], group["RSP Size"], group["Advertisement"],
                 group["RSSI promedio"]]))

            filtData = pd.DataFrame(data, columns=columns)
            filterData = pd.concat([filterData, filtData], ignore_index=True)

        path = Path("../results", tag)
        if not os.path.exists(path):
            path.mkdir(parents=True)

        path = Path(path, "ble-filter-clean-P_" + day + ".csv")
        filterData.to_csv(path, sep=';', index=False)

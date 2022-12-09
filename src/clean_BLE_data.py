import os
import pandas as pd
import numpy as np
from pathlib import Path


def cleanBLEData(dataPath, macList, sampling, tag):
    """Función que limpia los datos BLE brutos y guarda los resultados en un archivo CSV."""

    dataPath = Path(dataPath)
    macList = pd.read_csv(macList, delimiter=";", usecols=["MAC"])

    columns = ["Indice int. muestreo", "Timestamp int.", "Raspberry", "Timestamp inicial", "Nº Mensajes", "MAC",
               "Tipo MAC", "Tipo ADV", "BLE Size", "RSP Size", "BLE Data", "RSSI promedio"]

    for file in dataPath.iterdir():
        dataBle = pd.read_csv(file, sep=";")
        dataBle["Timestamp"] = pd.to_datetime(dataBle["Fecha"] + " " + dataBle["Hora"], format="%Y-%m-%d %H:%M:%S")
        initDate = dataBle["Timestamp"].min()
        endDate = dataBle["Timestamp"].max()

        fullDateList = list(enumerate(
            pd.date_range(start=dataBle["Fecha"].iloc[0] + " 07:00:00", end=dataBle["Fecha"].iloc[-1] + " 22:00:00",
                          freq=str(sampling) + "T")))

        dateList = np.array([x for x in fullDateList if initDate <= x[1] <= endDate])
        filterData = pd.DataFrame(columns=columns)
        for i, date in dateList:

            nextDate = [x for x in dateList if x[0] == i + 1]
            group = dataBle.loc[(dataBle["Timestamp"] >= date) & (dataBle["Timestamp"] < nextDate[0][1])]
            filtData = pd.DataFrame(columns=columns)

            for index, row in group.iterrows():
                print(macList["MAC"] in row["MAC"])
                if not (macList["MAC"] == row["MAC"]).any():
                    if ((filtData["Raspberry"] == row["Id"]) & (filtData["MAC"] == row["MAC"]) & (
                            filtData["BLE Data"] == row["Advertisement"])).any():

                        index = filtData.loc[
                            (filtData["Raspberry"] == row["Id"]) & (filtData["MAC"] == row["MAC"]) & (
                                    filtData['BLE Data'] == row['Advertisement'])].index[0]

                        filtData.at[index, 'Nº Mensajes'] += 1
                        filtData.at[index, 'RSSI promedio'] += row['RSSI']

                    else:
                        data = [i, date, row["Id"], row["Fecha"] + " " + row["Hora"], 1, row["MAC"], row["Tipo MAC"],
                                row["Tipo ADV"], row["ADV Size"], row["RSP Size"], row["Advertisement"], row["RSSI"]]

                        filtData = pd.concat([filtData, pd.DataFrame([data], columns=columns)], ignore_index=True)

            filtData['RSSI promedio'] = filtData['RSSI promedio'] / filtData['Nº Mensajes']
            filterData = pd.concat([filterData, filtData], ignore_index=True)

        path = Path("../results", tag)
        if not os.path.exists(path):
            path.mkdir(parents=True)

        path = Path(path, "ble-filter-clean-P_" + dataBle["Fecha"].iloc[0] + ".csv")
        filterData.to_csv(path, sep=';', index=False)

#!/usr/bin/python
import numpy as np
import pandas as pd
import time

sampling = 5
objetivo = "2022-07-06"
nombre_target = "./docs/data/ble_filter_" + objetivo + "_samp" + str(sampling) + ".csv"
nombre_filter = "./docs/data/ble-filter-clean-P_" + objetivo + ".csv"

# nombre_lista = "/home/servidoridiit1upct/CRAI-Servidor/mac_filter.csv"
nombre_lista = "./docs/mac_filter.csv"
filter_cols = ['Indice int. muestreo', 'Timestamp int.', 'Raspberry', 'Timestamp inicial', 'Nº Mensajes', 'MAC',
               'Tipo MAC', 'Tipo ADV', 'BLE Size', 'RSP Size', 'BLE Data', 'RSSI promedio']

lista_filtro = pd.read_csv(nombre_lista, delimiter=';')
datosafiltrar = pd.read_csv(nombre_target, delimiter=';')

datos_filtrados = pd.DataFrame(columns=filter_cols)
datos_filtrados.to_csv(nombre_filter, sep=';', index=False, mode='w')
hora = 7
go = True
while go:

    desde_hora = objetivo + " " + str(hora).zfill(2) + ":00:00"
    hasta_hora = objetivo + " " + str(hora + 1).zfill(2) + ":00:00"
    datos_intervalo = datosafiltrar[datosafiltrar['Timestamp int.'] >= desde_hora]

    datos_intervalo = datos_intervalo[datos_intervalo['Timestamp int.'] < hasta_hora]
    print(f"Filtrando en intervalo: {desde_hora} - {hasta_hora}")

    for index, row in datos_intervalo.iterrows():
        if not (lista_filtro['MAC'] == row['MAC']).any():
            # print(f"Intervalo cursado: {row['Timestamp int.']}")
            datos_filtrados = pd.concat([datos_filtrados, pd.DataFrame([row.values], columns=filter_cols)], ignore_index=True, axis=0)

    datos_filtrados.to_csv(nombre_filter, sep=';', index=False, mode='a', header=False)
    hora += 1
    if hora == 23:
        go = False
    datos_filtrados = pd.DataFrame(columns=filter_cols)

print("End of filtering")

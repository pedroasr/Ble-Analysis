import pandas as pd
import time

# Parámetros

sampling = 5  # Periodo de muestreo

if sampling > 60:
    print("Max sampling period is 60!")
    exit()
# ---------------
temp = 60 - sampling
final = (60 * 15) / sampling

objetivo = "2022-07-05"
print("Filtering BLE Data this is going to last some time")

hora_inicio = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

print(f'Hora inicio: {hora_inicio}')
# Variable funcionamiento

# Para localizar los CSV
nombre_target = "./docs/data/ble_" + objetivo + "_7-22.csv"  # Nombre del archivo
nombre_filter = "./docs/data/ble_filter_" + objetivo + "_samp" + str(sampling) + ".csv"
filter_cols = ['Indice int. muestreo', 'Timestamp int.', 'Raspberry', 'Timestamp inicial', 'Nº Mensajes', 'MAC',
               'Tipo MAC', 'Tipo ADV', 'BLE Size', 'RSP Size', 'BLE Data', 'RSSI promedio']
nseq = 0

datos_ble = pd.read_csv(nombre_target, sep=';')

desde_hora = 7
hasta_hora = 7

desde = 0
hasta = desde + sampling

if sampling == 60:
    hasta_hora = 8
    hasta = 0

go = True

while go:

    desde_tiempo = ''
    hasta_tiempo = ''

    desde_tiempo += str(desde_hora).zfill(2) + ":" + str(desde).zfill(2) + ":00"
    hasta_tiempo += str(hasta_hora).zfill(2) + ":" + str(hasta).zfill(2) + ":00"

    if desde == temp:
        desde = 0
        desde_hora += 1
    else:
        desde += sampling

    if hasta == temp:
        hasta = 0
        hasta_hora += 1
    else:
        hasta += sampling

    nseq += 1

    aux = datos_ble.loc[datos_ble["Hora"] >= desde_tiempo]
    aux = aux.loc[aux["Hora"] < hasta_tiempo]

    datos_filtrados = pd.DataFrame(columns=filter_cols)

    # Start filtering
    for index, row in aux.iterrows():
        if ((datos_filtrados['Raspberry'] == row['Id']) & (datos_filtrados['MAC'] == row['MAC']) & (datos_filtrados['BLE Data'] == row['Advertisement'])).any():

            indice = datos_filtrados.loc[
                (datos_filtrados['Raspberry'] == row['Id']) & (datos_filtrados['MAC'] == row['MAC']) & (
                    datos_filtrados['BLE Data'] == row['Advertisement'])].index[0]

            datos_filtrados.at[indice, 'Nº Mensajes'] += 1
            datos_filtrados.at[indice, 'RSSI promedio'] += row['RSSI']

        else:
            data = [nseq, objetivo + " " + desde_tiempo, row['Id'],
                    row['Fecha'] + " " + row['Hora'], 1, row['MAC'], row['Tipo MAC'], row['Tipo ADV'], row['ADV Size'],
                    row['RSP Size'], row['Advertisement'], row['RSSI']]
            datos_filtrados = pd.concat([datos_filtrados, pd.DataFrame([data], columns=filter_cols)], ignore_index=True)

    # Now save in csv
    datos_filtrados['RSSI promedio'] = datos_filtrados['RSSI promedio'] / datos_filtrados['Nº Mensajes']

    if nseq == 1:
        print("first time")
        datos_filtrados.to_csv(nombre_filter, sep=';', index=False)
    else:
        print()
        datos_filtrados.to_csv(nombre_filter, sep=';', mode='a', header=False, index=False)

    print(f'Intervalos escritos: {nseq}/{final}')

    if desde_hora == 22:
        go = False

hora_fin = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
print(f'Filtrado acabado, hora: {hora_fin}')

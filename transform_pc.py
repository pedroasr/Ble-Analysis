import pandas as pd

data = pd.read_csv("../docs/personcount_test/pcount_2022-10-20.csv", sep=";")

data["Ocupacion"] = 0
data["Estado"] = 1

data['Timestamp'] = data["Fecha"] + " " + data["Hora"]

data.replace({"Right2": "Right"}, inplace=True)
data.drop_duplicates(subset=['Timestamp', 'Sensor'], keep='first', inplace=True)
data = data.drop(data[data["Sensor"] == "KeepAlive"].index)
data.reset_index(drop=True, inplace=True)
for i in range(len(data)):
    if i == 0:
        if data["Evento In-Out(1/0)"][i] == 1:
            data["Ocupacion"][i] = data["Ocupacion"][i] + 1
        else:
            data["Ocupacion"][i] = data["Ocupacion"][i] - 1
    else:
        if data["Evento In-Out(1/0)"][i] == 1:
            data["Ocupacion"][i] = data["Ocupacion"][i - 1] + 1
        else:
            data["Ocupacion"][i] = data["Ocupacion"][i - 1] - 1

data.to_csv("../docs/personcount_test/pcount_filter_2022-10-20.csv", sep=";", index=False)

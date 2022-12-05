import pandas as pd

data = pd.read_csv("../docs/personcount_test/pc_2022-09-26.csv", sep=";")

data["Estado"] = 1

data['Timestamp'] = data["Fecha"] + " " + data["Hora"]

data.replace({"Right2": "Right"}, inplace=True)
data.drop_duplicates(subset=['Timestamp', 'Sensor'], keep='first', inplace=True)
data = data.drop(data[data["Sensor"] == "KeepAlive"].index)
data.reset_index(drop=True, inplace=True)
data["Ocupacion"] = (2 * data["Evento In-Out(1/0)"].astype(int) - 1).cumsum()

data.to_csv("../docs/personcount_test/pcount_filter_2022-09-26_prueba.csv", sep=";", index=False)

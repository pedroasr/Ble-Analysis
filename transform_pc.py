import pandas as pd
import numpy as np

data = pd.read_csv("../docs/pc_2022-10-06.csv", sep=";")

data["Ocupacion"] = 0

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

data.to_csv("../docs/pc_2022-10-06_prueba.csv", sep=";", index=False)


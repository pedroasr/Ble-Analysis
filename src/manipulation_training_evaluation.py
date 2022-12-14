from clean_ble_data import cleanBLEData
from manipulate_ble_data import readAndPrepareDataFromDirectory, getDataset
from train_models import trainModels
from val_models import valModels

sampling = 5

# Se limpian los datos en bruto, generando archivos csv en la carpeta results.
cleanBLEData("../data/ble_learning", "../data/mac_filter.csv", sampling, "ble_learning")

cleanBLEData("../data/ble_validation", "../data/mac_filter.csv", sampling, "ble_validation")

# Se cargan los datos de entrenamiento y test.
dataListTrain, personCountListTrain, stateListTrain = readAndPrepareDataFromDirectory("../results/ble_learning",
                                                                                      "../data/personcount_learning",
                                                                                      "../data/state_learning",
                                                                                      sampling)

dataListVal, personCountListVal, stateListVal = readAndPrepareDataFromDirectory("../results/ble_validation",
                                                                                "../data/personcount_validation",
                                                                                "../data/state_validation", sampling)

# Se procesan los datos hasta obtener los conjuntos de entrenamiento y test.
getDataset(dataListTrain, personCountListTrain, stateListTrain, "learning", "Full", "Training", sampling)

getDataset(dataListVal, personCountListVal, stateListVal, "validation", "Full", "Training", sampling)

# Se entrenan los modelos y se evaluan los resultados.
trainModels("../results/learning/filled-learning-set.csv")

valModels("../results/validation/filled-validation-set.csv", "../models/ExtraTreesRegressor.pkl",
          "../models/XGBRegressor.pkl", "../models/RandomForestRegressor.pkl", "Validation")

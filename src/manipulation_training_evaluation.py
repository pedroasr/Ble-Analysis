from clean_BLE_data import cleanBLEData
from functions_off import readAndPrepareDataFromDirectory, getDataset
from train_models import trainModels
from test_models import testModels

sampling = 5

# Se limpian los datos en bruto, generando archivos csv en la carpeta results.
cleanBLEData("../data/ble", "../data/mac_filter.csv", sampling, "ble")

cleanBLEData("../data/ble_test", "../data/mac_filter.csv", sampling, "ble_test")

# Se cargan los datos de entrenamiento y test.
dataListTrain, personCountListTrain, stateListTrain = readAndPrepareDataFromDirectory("../results/ble",
                                                                                      "../data/personcount",
                                                                                      "../data/state", sampling)

dataListTest, personCountListTest, stateListTest = readAndPrepareDataFromDirectory("../results/ble_test",
                                                                                   "../data/personcount_test",
                                                                                   "../data/state_test", sampling)

# Se procesan los datos hasta obtener los conjuntos de entrenamiento y test.
getDataset(dataListTrain, personCountListTrain, stateListTrain, "training", "../results", "Initial", "Final", sampling)

getDataset(dataListTest, personCountListTest, stateListTest, "test", "../results", "Initial", "Final", sampling)

# Se entrenan los modelos y se evaluan los resultados.
trainModels("../results/training/filled-training-set.csv", "../models")

testModels("../results/test/filled-test-set.csv", "../models/ExtraTreesRegressor.pkl", "../models/XGBRegressor.pkl",
           "../models/RandomForestRegressor.pkl", "Test")

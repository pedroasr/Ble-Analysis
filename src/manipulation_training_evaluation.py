from functions_off import readAndPrepareDataFromDirectory, getDataset
from train_models import trainModels
from test_models import testModels


# Se cargan los datos de entrenamiento y test.
dataListTrain, personCountListTrain, stateListTrain = readAndPrepareDataFromDirectory("../data/ble",
                                                                                      "../data/personcount",
                                                                                      "../data/state")

dataListTest, personCountListTest, stateListTest = readAndPrepareDataFromDirectory("../data/ble_test",
                                                                                   "../data/personcount_test",
                                                                                   "../data/state_test")

# Se procesan los datos hasta obtener los conjuntos de entrenamiento y test.
getDataset(dataListTrain, personCountListTrain, stateListTrain, "training", "../results", "Initial", "Final")

getDataset(dataListTest, personCountListTest, stateListTest, "test", "../results", "Initial", "Final")

# Se entrenan los modelos y se evaluan los resultados.
trainModels("../results/training/filled-training-set.csv", "../models")

testModels("../results/test/filled-test-set.csv", "../models/ExtraTreesRegressor.pkl", "../models/XGBRegressor.pkl",
           "../models/RandomForestRegressor.pkl", "Test")

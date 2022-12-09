from functions_off import readAndPrepareDataFromDirectory, getDataset
from train_models import trainModels
from test_models import testModels


# Se cargan los datos de entrenamiento y test.
dataListTrain, personCountListTrain, stateListTrain = readAndPrepareDataFromDirectory("../docs/data",
                                                                                      "../docs/personcount",
                                                                                      "../docs/state")

dataListTest, personCountListTest, stateListTest = readAndPrepareDataFromDirectory("../docs/data_test",
                                                                                   "../docs/personcount_test",
                                                                                   "../docs/state_test")

# Se procesan los datos hasta obtener los conjuntos de entrenamiento y test.
getDataset(dataListTrain, personCountListTrain, stateListTrain, "training", "../docs/prueba", "../figures/prueba", "../figuresFilled/prueba")

getDataset(dataListTest, personCountListTest, stateListTest, "test", "../docs/prueba", "../figures/prueba", "../figuresFilled/prueba")

# Se entrenan los modelos y se evaluan los resultados.
trainModels("../docs/prueba/filled-training-set.csv", "../models/prueba")

testModels("../docs/prueba/filled-test-set.csv", "../models/prueba/ExtraTreesRegressor.pkl", "../models/prueba/XGBRegressor.pkl",
           "../models/prueba/RandomForestRegressor.pkl", "../figuresTest/prueba")

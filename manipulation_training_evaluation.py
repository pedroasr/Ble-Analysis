from functions_off import readDataFromDirectory, getDataset
from train_models import trainModels
from test_models import testModels

# Se cargan los datos de entrenamiento y test.
dataListTrain, personCountListTrain, stateListTrain = readDataFromDirectory("../docs/data", "../docs/personcount",
                                                                            "../docs/state")

dataListTest, personCountListTest, stateListTest = readDataFromDirectory("../docs/data_test",
                                                                         "../docs/personcount_test",
                                                                         "../docs/state_test")

# Se procesan los datos hasta obtener los conjuntos de entrenamiento y test.
getDataset(dataListTrain, personCountListTrain, stateListTrain, "training")

getDataset(dataListTest, personCountListTest, stateListTest, "test")

# Se entrenan los modelos y se evaluan los resultados.
trainModels("../docs/filled-training-set.csv", "../docs")

testModels("../docs/filled-test-set.csv", "../models/ExtraTreesRegressor.pkl", "../models/XGBRegressor.pkl",
           "../models/RandomForestRegressor.pkl", "../figuresTest")

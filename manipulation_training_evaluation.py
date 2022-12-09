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
getDataset(dataListTrain, personCountListTrain, stateListTrain, "training", "../docs", "../figures", "../figuresFilled")

getDataset(dataListTest, personCountListTest, stateListTest, "test", "../docs", "../figures", "../figuresFilled")

# Se entrenan los modelos y se evaluan los resultados.
trainModels("../docs/filled-training-set.csv", "../models")

testModels("../docs/filled-test-set.csv", "../models/ExtraTreesRegressor.pkl", "../models/XGBRegressor.pkl",
           "../models/RandomForestRegressor.pkl", "../figuresTest")

from functions_op import *
import pandas as pd
# import joblib
import warnings

warnings.filterwarnings("ignore")

pd.options.mode.chained_assignment = None

dataArray, personCountArray, stateArray = readDataFromDirectory("../docs/data_prueba", "../docs/personcount_prueba",
                                                                "../docs/state_prueba")

trainingDataset, filterDataSet = getTrainingDataset(dataArray, personCountArray, stateArray)

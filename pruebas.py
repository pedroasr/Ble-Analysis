from functions_op import *
import pandas as pd
# import joblib
import warnings

warnings.filterwarnings("ignore")

pd.options.mode.chained_assignment = None

dataArray, personCountArray, stateArray = readDataFromDirectory("../docs/data", "../docs/personcount", "../docs/state")

trainingDataset, filterDataSet, filledDataset = getTrainingDataset(dataArray, personCountArray, stateArray)

"""
 Author: Niharika gauraha
 Implementation of CCP
"""

from sklearn.model_selection import train_test_split
import random
import RegressionICP as icp
import perf_measure as pm
import numpy as np
import matplotlib.pylab as plt
from prettytable import PrettyTable
from sklearn.model_selection import ShuffleSplit
from perf_measure import pValues2PerfMetrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import OrderedDict
import dataset_preprocessing as data

iteration = 10

def ACP(X, y, n_source = 3, method = 'rf', file =None):
    effList = []

    for i in range(iteration):
        XX, X_test, yy, y_test \
            = train_test_split(X, y, test_size=0.2, random_state=i)

        #scaler = StandardScaler()
        scaler = MinMaxScaler()
        scaler.fit(XX)
        XX = scaler.transform(XX)
        X_test = scaler.transform(X_test)

        scaler = MinMaxScaler()
        scaler.fit(yy.reshape(-1, 1))
        yy = scaler.transform(yy.reshape(-1, 1)).ravel()
        y_test = scaler.transform(y_test.reshape(-1, 1)).ravel()

        intervals = np.zeros((n_source, len(y_test), 2))
        sourceIndex = 0
        #sss = StratifiedShuffleSplit(n_splits=n_source, test_size=1/n_source)
        sss = ShuffleSplit(n_splits=n_source, test_size=1/n_source, random_state=i)
        for train_index, test_index in sss.split(XX, yy):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_calib = XX[train_index], XX[test_index]
            y_train, y_calib = yy[train_index], yy[test_index]
            intervals[sourceIndex,:,:], testPred = icp.ICPRegression(X_train, y_train,
                                                                     X_calib, y_calib,
                                                                     X_test,
                                                                     method = method)
            #eff, errRate = pValues2PerfMetrics(intervals[sourceIndex], y_test)
            sourceIndex = sourceIndex+1
            #print(eff, errRate)

        combined_intervals = aggregateIntervals(intervals)
        #combined_intervals = combine_intervals(intervals)
        eff, errRate = pValues2PerfMetrics(combined_intervals, y_test)
        effList.append(eff)
        print(i)
    import os
    if not os.path.exists('json_acp'):
        os.makedirs('json_acp')

    import json
    file_name = "json_acp/" + file + method+".json"
    with open(file_name, 'w') as fh:
        fh.write(json.dumps(effList))

    return effList


#combine intervals with smallest and largest values
def combine_intervals(intervals):
    low = np.min(intervals[:,:,0], axis=0)
    high = np.max(intervals[:,:,1],axis=0)
    return np.column_stack((low, high))


def aggregateIntervals(intervals):
    low = np.median( intervals[:, : ,0], axis=0)
    high = np.median(intervals[:, :, 1], axis=0)
    return np.column_stack((low, high))


if __name__ == '__main__':
    nrSources = 3
    method = "rf"

    dataset_names = ['Housing', 'Wine', 'PD', 'PowerPlant', 'Energy', 'Concrete',
                     'GridStability', 'SuperConduct', 'CBM', 'Game']

    load_functions = OrderedDict()
    load_functions["Housing"] = data.load_boston_data
    load_functions["PD"] = data.load_PD_data
    load_functions["PowerPlant"] = data.load_powerPlant_data
    load_functions["Energy"] = data.load_energy_data
    load_functions["Concrete"] = data.load_concrete_data
    load_functions["CBM"] = data.load_CBM_data
    load_functions["Game"] = data.load_game_data
    load_functions["Wine"] = data.load_wine_data
    load_functions["GridStability"] = data.load_gridStability_data
    load_functions["SuperConduct"] = data.load_superConduct_data


    for dataset_name in dataset_names:
        X, y = load_functions[dataset_name]()

        print(dataset_name)
        ACP(X, y, n_source=nrSources ,method = method, file=dataset_name)



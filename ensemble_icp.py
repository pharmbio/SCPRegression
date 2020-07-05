"""
 Author: Niharika gauraha
 Synergy Conformal Prediction
"""

import numpy as np
from sklearn.model_selection import train_test_split
import RegressionICP as icp
from perf_measure import pValues2PerfMetrics
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dataset_preprocessing import equal_partition
from sklearn.model_selection import ShuffleSplit


epsilon = .1
iteration = 10

# Limitations: This will work for only three sources
# For many partitions see manyPartitions.py
def ensembleICP(X, y, n_source = 3, methods = None, path = None):
    listICPEff = []  # empty list

    if methods is None:
        methods = ['linear_svr'] * n_source

    for i in range(iteration):

        X_part, X_test, y_part, y_test \
            = train_test_split(X, y, test_size=0.2, random_state=i)

        scaler = MinMaxScaler()
        scaler.fit(X_part)
        X_part = scaler.transform(X_part)
        X_test = scaler.transform(X_test)

        # scaling of Output variable
        scaler = MinMaxScaler()
        scaler.fit(y_part.reshape(-1, 1))
        y_part = scaler.transform(y_part.reshape(-1, 1)).ravel()
        y_test = scaler.transform(y_test.reshape(-1, 1)).ravel()

        # keep aside the calib set which is not used in this case
        X_part, X_calib, y_part, y_calib \
            = train_test_split(X_part, y_part, test_size=0.3, random_state=i)

        intervals = np.zeros((n_source, len(y_test), 2))
        indexSrc = 0
        sss = ShuffleSplit(n_splits=n_source, test_size=1 / n_source, random_state=i)

        for train_index, test_index in sss.split(X_part, y_part):
            X_train, X_calib = X_part[train_index], X_part[test_index]
            y_train, y_calib = y_part[train_index], y_part[test_index]

            intervals[indexSrc,:,:], testPred = icp.ICPRegression(X_train, y_train,
                                                     X_calib, y_calib, X_test,
                                                     method=methods[indexSrc],
                                                     nrTrees=10)


            indexSrc += 1

        combined_intervals = aggregateIntervals(intervals)
        # combined_intervals = combine_intervals(intervals)
        eff, errRate = pValues2PerfMetrics(combined_intervals, y_test)
        listICPEff.append(eff)
        print(i)

    results = OrderedDict()
    results["enICP"] = listICPEff

    # path should exist, it will not check for the same
    import json
    with open(path, 'w') as fh:
        fh.write(json.dumps(results))

    print(path, round(np.median(listICPEff), 3))


def aggregateIntervals(intervals):
    low = np.median( intervals[:, : ,0], axis=0)
    high = np.median(intervals[:, :, 1], axis=0)
    return np.column_stack((low, high))

#just to compute the partition size
def computeSize(X, y):
    X_train, X_test, y_train, y_test \
        = train_test_split(X, y, test_size=0.2)
    X_train, X_calib, y_train, y_calib \
        = train_test_split(X_train, y_train, test_size=0.3)
    print(len(X_train), len(X_calib), len(X_test), X_train.shape[1])


if __name__=="__main__":
    import dataset_preprocessing as data

    X, y = data.load_boston_data()
    computeSize(X, y)
    import os
    if not os.path.exists('json'):
        os.makedirs('json')

    file_name = "json/" + "temp.json"
    ensembleICP(X, y, n_source=3, path=file_name)




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


epsilon = .1
iteration = 10

# Limitations: This will work for only three sources
# For many partitions see manyPartitions.py
def synergyCP(X, y, n_source = 3, methods = None, path = None):
    listSmallICPEff = []  # empty list
    for i in range(n_source):
        listSmallICPEff.append([])

    listSCPEff = []  # empty list

    if methods is None:
        methods = ['linear_svr'] * n_source

    for i in range(iteration):

        X_train, X_test, y_train, y_test \
            = train_test_split(X, y, test_size=0.2, random_state=i)

        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # scaling of Output variable
        scaler = MinMaxScaler()
        scaler.fit(y_train.reshape(-1, 1))
        y_train = scaler.transform(y_train.reshape(-1, 1)).ravel()
        y_test = scaler.transform(y_test.reshape(-1, 1)).ravel()

        X_train, X_calib, y_train, y_calib \
            = train_test_split(X_train, y_train, test_size=0.3, random_state=i)


        meanCalibConfScore = np.zeros(len(y_calib))
        meanTestPred = np.zeros(len(y_test))

        list_part = equal_partition(len(X_train), N=3, seed=i)

        for indexSrc in range(0, n_source):
            part_index = list_part[indexSrc]
            sourceData = X_train[part_index]
            sourceTarget = y_train[part_index]
            calib_pred, testPred = icp.ICPRegression(sourceData, sourceTarget,
                                                     X_calib, y_calib, X_test,
                                                     method=methods[indexSrc],
                                                     returnPredictions=True,
                                                     nrTrees=10)
            confScores = icp.computeConformityScores(calib_pred, y_calib)
            intervals = icp.computeInterval(np.sort(confScores), testPred, epsilon)
            eff, errRate = pValues2PerfMetrics(intervals, y_test)
            listSmallICPEff[indexSrc].append(eff)

            meanCalibConfScore = np.add(meanCalibConfScore, calib_pred)
            meanTestPred = np.add(meanTestPred, testPred)

        meanCalibConfScore = meanCalibConfScore / n_source
        meanTestPred = meanTestPred / n_source
        meanCalibConfScore = icp.computeConformityScores(meanCalibConfScore, y_calib)
        intervals = icp.computeInterval(np.sort(meanCalibConfScore), meanTestPred, epsilon)
        eff, errRate = pValues2PerfMetrics(intervals, y_test)
        #print(errRate)
        listSCPEff.append(eff)
        print(i)

    results = OrderedDict()
    results["source1"] = listSmallICPEff[0]
    results["source2"] = listSmallICPEff[1]
    results["source3"] = listSmallICPEff[2]
    results["SCP"] = listSCPEff

    # path should exist, it will not check for the same
    import json
    with open(path, 'w') as fh:
        fh.write(json.dumps(results))

    print(path, round(np.median(listSmallICPEff[0]), 3),
               round(np.median(listSmallICPEff[1]), 3),
               round(np.median(listSmallICPEff[2]), 3),
               round(np.median(listSCPEff), 3))



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
    synergyCP(X, y, n_source=3, path=file_name)




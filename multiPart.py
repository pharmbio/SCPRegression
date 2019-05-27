"""
 Author: Niharika gauraha
 Synergy conformal prediction using linear SVM classifier
"""
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split
import random
from prettytable import PrettyTable
from dataset_preprocessing import *
import RegressionICP as icp
from perf_measure import pValues2PerfMetrics
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler, MinMaxScaler

boxPlot = False
x = PrettyTable()
#x.field_names = ["Dataset", "smallICP_1", "smallICP_2", "smallICP_3", "ICP", "SCP"]
x.field_names = ["Dataset", "smallICP_$p$", "SCP"]
epsilon = 0.1
iteration = 10

def synergyCP(X, y, n_source = 3, file = None):
    listSmallICPEff = []  # empty list

    for i in range(n_source):
        listSmallICPEff.append([])

    listSCPEff = []  # empty list

    for i in range(iteration):
        X_train, X_test, y_train, y_test \
            = train_test_split(X, y, test_size=0.2, random_state=i)

        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        scaler = MinMaxScaler()
        scaler.fit(y_train.reshape(-1, 1))
        y_train = scaler.transform(y_train.reshape(-1, 1)).ravel()
        y_test = scaler.transform(y_test.reshape(-1, 1)).ravel()

        X_train, X_calib, y_train, y_calib \
            = train_test_split(X_train, y_train, test_size=0.3, random_state=i)

        nrTrainCases = len(y_train)
        randIndex = random.sample(list(range(0, nrTrainCases)), nrTrainCases)
        splitLen = int(nrTrainCases / n_source)
        # split training data into equal parts
        trainIndex = randIndex[0:splitLen]


        meanCalibConf = np.zeros(len(y_calib))
        meanTestPred = np.zeros(len(y_test))

        for indexSrc in range(0, n_source):
            sourceData = X_train[trainIndex, :]
            sourceTarget = y_train[trainIndex]
            calib_pred, testPred = icp.ICPRegression(sourceData, sourceTarget,
                                                     X_calib, y_calib, X_test,
                                                     method=method, returnPredictions=True)
            confScores = icp.computeConformityScores(calib_pred, y_calib)
            intervals = icp.computeInterval(np.sort(confScores), testPred, epsilon)
            eff, errRate = pValues2PerfMetrics(intervals, y_test)
            listSmallICPEff[indexSrc].append(eff)

            meanCalibConf = np.add(meanCalibConf, calib_pred)
            meanTestPred = np.add(meanTestPred, testPred)

            trainIndex = randIndex[splitLen * (indexSrc + 1):splitLen * (indexSrc + 2)]

        meanCalibConf = meanCalibConf / n_source
        meanTestPred = meanTestPred / n_source
        meanCalibConf = icp.computeConformityScores(meanCalibConf, y_calib)
        intervals = icp.computeInterval(np.sort(meanCalibConf), meanTestPred, epsilon)
        eff, errRate = pValues2PerfMetrics(intervals, y_test)
        listSCPEff.append(eff)

    x.add_row([file, round(min(np.mean(np.array(listSmallICPEff), axis=1)),3),
               round(np.mean(listSCPEff),3)])

    results = OrderedDict()
    results["ICP_p"] = listSmallICPEff
    results["SCP"] = listSCPEff

    return results



if __name__ == '__main__':
    nrSources = [5, 7, 9, 11, 15, 20]
    #nrSources = [5, 7]
    method = 'svr'
    LogData = OrderedDict()
    dataset_name = "SuperConduct"
    load_function = load_superConduct_data
    dataset_name = "Housing"
    load_function = load_boston_data

    for i in range(len(nrSources)):
        X, y = load_function()
        results = synergyCP(X, y, n_source=nrSources[i], file=dataset_name)
        LogData[str(nrSources[i])] = results

    import os

    if not os.path.exists('json_multi'):
        os.makedirs('json_multi')

    import json
    file_name = "json_multi/"+dataset_name + ".json"
    with open(file_name, 'w') as fh:
        fh.write(json.dumps(LogData))


    print(x)




"""
 Author: Niharika gauraha
 Synergy Conformal Prediction using whole data and
 three different methods
"""
from prettytable import PrettyTable
from dataset_preprocessing import *
from sklearn.model_selection import train_test_split
import numpy as np
import RegressionICP as icp
from perf_measure import pValues2PerfMetrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import OrderedDict

x = PrettyTable()
x.field_names = ["Dataset", "SVR-ICP", "RF-ICP", "RBF-SVR-ICP", 'SCP']
epsilon = 0.1
iterate = 10

methods = ["linear_svr", "rf", "svr"]

def synergyCP(X, y, methods = None, file = None):
    n_source = len(methods)
    listIndSrcEff = []  # empty list
    for i in range(n_source):
        listIndSrcEff.append([])

    listSCPEff = []  # empty list

    for i in range(iterate):
        X_train, X_test, y_train, y_test \
            = train_test_split(X, y, test_size=0.2, random_state=i)

        #scaler = StandardScaler()
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

        meanCalibConfScore = np.zeros(len(y_calib))
        meanTestPred = np.zeros(len(y_test))
        for indexSrc in range(n_source):
            sourceData = X_train
            sourceTarget = y_train
            calib_pred, testPred = icp.ICPRegression(sourceData, sourceTarget,
                                                     X_calib, y_calib, X_test,
                                                     method=methods[indexSrc],
                                                     returnPredictions=True,
                                                     nrTrees=10)
            confScores = icp.computeConformityScores(calib_pred, y_calib)
            intervals = icp.computeInterval(np.sort(confScores), testPred, epsilon)
            eff, errRate = pValues2PerfMetrics(intervals, y_test)
            listIndSrcEff[indexSrc].append(eff)


            meanCalibConfScore = np.add(meanCalibConfScore, calib_pred)
            meanTestPred = np.add(meanTestPred, testPred)

        meanCalibConfScore = meanCalibConfScore / n_source
        meanTestPred = meanTestPred / n_source
        meanCalibConfScore = icp.computeConformityScores(meanCalibConfScore, y_calib)
        intervals = icp.computeInterval(np.sort(meanCalibConfScore), meanTestPred, epsilon)
        eff, errRate = pValues2PerfMetrics(intervals, y_test)
        listSCPEff.append(eff)
        print(i)

    results = OrderedDict()
    results["source1"] = listIndSrcEff[0]
    results["source2"] = listIndSrcEff[1]
    results["source3"] = listIndSrcEff[2]
    results["SCP"] = listSCPEff
    import os
    if not os.path.exists('json_same_data'):
        os.makedirs('json_same_data')

    import json
    file_name = "json_same_data/" + file + ".json"
    with open(file_name, 'w') as fh:
        fh.write(json.dumps(results))



    #listACPEff= [0,1,2]
    '''
    x.add_row([file, round(np.mean(listIndSrcEff[0]),3),
               round(np.mean(listIndSrcEff[1]), 3),
               round(np.mean(listIndSrcEff[2]), 3),
               round(np.mean(listSCPEff),3)])
    '''
    x.add_row([file, round(np.median(listIndSrcEff[0]), 3),
               round(np.median(listIndSrcEff[1]), 3),
               round(np.median(listIndSrcEff[2]), 3),
               round(np.median(listSCPEff), 3)])

if __name__ == '__main__':
    nrSources = 3

    dataset_names = ['Housing', 'PD', 'PowerPlant', 'Energy', 'Concrete', 'CBM', 'Game']
    dataset_names = ['Wine', 'PD', 'PowerPlant', 'Energy', 'Concrete',
                     'GridStability', 'SuperConduct', 'CBM', 'Game']

    load_functions = OrderedDict()
    load_functions["Housing"] = load_boston_data
    load_functions["PD"] = load_PD_data
    load_functions["PowerPlant"] = load_powerPlant_data
    load_functions["Energy"] = load_energy_data
    load_functions["Concrete"] = load_concrete_data
    load_functions["CBM"] = load_CBM_data
    load_functions["Game"] = load_game_data
    # these are time taking
    # dataset_names = ['Wine', 'GridStability', 'SuperConduct']
    load_functions["Wine"] = load_wine_data
    load_functions["GridStability"] = load_gridStability_data
    load_functions["SuperConduct"] = load_superConduct_data

    dataset_names = ['Concrete']

    for dataset_name in dataset_names:
        X, y = load_functions[dataset_name]()
        #listACPEff, listACPErr = acp.ACP(X, y)
        print(dataset_name)
        synergyCP(X, y, methods = methods, file=dataset_name)

    print(x)






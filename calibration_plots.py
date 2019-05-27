"""
 Author: Niharika gauraha
 Synergy Conformal Prediction Using Random Forest Classifier
"""

from sklearn.model_selection import train_test_split
import random
import dataset_preprocessing as data
import RegressionICP as icp
from perf_measure import pValues2PerfMetrics
import numpy as np
import matplotlib.pylab as plt
from prettytable import PrettyTable
from sklearn.model_selection import ShuffleSplit


def synergyCP(X, y, n_source = 3):


    X_train, X_test, y_train, y_test \
        = train_test_split(X, y, test_size=0.2)

    X_train, X_calib, y_train, y_calib \
        = train_test_split(X_train, y_train, test_size=0.3)

    nrTrainCases = len(y_train)
    randIndex = random.sample(list(range(0, nrTrainCases)), nrTrainCases)
    splitLen = int(nrTrainCases / n_source)
    # split training data into equal parts
    trainIndex = randIndex[0:splitLen]

    # whole ICP
    #calib_pred, testPred = icp.ICPRegression(X_train, y_train,
    #                                         X_calib, y_calib, X_test,
    #                                         method="rf", returnPredictions=True, nrTrees=10)

    meanCalibPred = np.zeros(len(y_calib))
    meanTestPred = np.zeros(len(y_test))

    for indexSrc in range(0, n_source):
        sourceData = X_train[trainIndex, :]
        sourceTarget = y_train[trainIndex]
        calib_pred, testPred = icp.ICPRegression(sourceData, sourceTarget,
                                                 X_calib, y_calib, X_test,
                                                 method="rf", returnPredictions=True, nrTrees=10)
        #confScores = icp.computeConformityScores(calib_pred, y_calib)
        #intervals = icp.computeInterval(confScores, testPred, epsilon)
        #eff, errRate = pValues2PerfMetrics(intervals, y_test)
        #listSmallICPEff[indexSrc].append(eff)

        meanCalibPred = np.add(meanCalibPred, calib_pred)
        meanTestPred = np.add(meanTestPred, testPred)
        '''
        sigLevels = np.linspace(0.01, .99, 100)
        errorRate = np.zeros(len(sigLevels))
        for i in range(len(sigLevels)):
            confScores = icp.computeConformityScores(calib_pred, y_calib)
            intervals = icp.computeInterval(confScores, testPred, sigLevels[i])
            eff, errorRate[i] = pValues2PerfMetrics(intervals, y_test)

        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.plot(sigLevels, errorRate, color="orange")
        plt.annotate("",
                     xy=(0, 0), xycoords='data',
                     xytext=(1, 1), textcoords='data',
                     arrowprops=dict(arrowstyle="-",
                                     connectionstyle="arc3,rad=0.",
                                     color='r')
                     )
        plt.show()
        '''
        trainIndex = randIndex[splitLen * (indexSrc + 1):splitLen * (indexSrc + 2)]

    meanCalibPred = meanCalibPred / n_source
    meanTestPred = meanTestPred / n_source
    sigLevels = np.linspace(0.01, .99, 100)
    errorRate = np.zeros(len(sigLevels))
    for i in range(len(sigLevels)):
        confScores = icp.computeConformityScores(meanCalibPred, y_calib)
        intervals = icp.computeInterval(confScores, meanTestPred, sigLevels[i])
        eff, errorRate[i] = pValues2PerfMetrics(intervals, y_test)

    plt.xlim((0, 1))
    plt.ylim((0,1))
    plt.plot(sigLevels, errorRate, color="blue")
    plt.annotate("",
                 xy=(0, 0), xycoords='data',
                 xytext=(1, 1), textcoords='data',
                 arrowprops=dict(arrowstyle="-",
                                 connectionstyle="arc3,rad=0.",
                                 color='r')
                 )
    plt.show()




def combine_intervals(intervals):
    low = np.min(intervals[:,:,0], axis=0)
    high = np.max(intervals[:,:,1],axis=0)
    return np.column_stack((low, high))


# compute median width and median center point
def compute_intervals(intervals):
    width = np.median( np.abs(intervals[:,:,0] - intervals[:,:,1]), axis=0)
    median = np.median((intervals[:, :, 1] + intervals[:, :, 0])/2, axis=0)
    low = median - width/2
    high = median + width / 2
    return np.column_stack((low, high))

# compute means of the lower and upper bounds
def aggregateIntervals(intervals):
    low = np.median( intervals[:, : ,0], axis=0)
    high = np.median(intervals[:, :, 1], axis=0)
    return np.column_stack((low, high))


def ACP(X, y, n_source = 3):
    effList = []
    errRateList = []

    XX, X_test, yy, y_test \
        = train_test_split(X, y, test_size=0.2)

    sigLevels = np.linspace(0.01, .99, 100)
    errorRate = np.zeros(len(sigLevels))

    intervals = np.zeros((len(sigLevels), n_source, len(y_test), 2))
    sourceIndex = 0

    sss = ShuffleSplit(n_splits=n_source, test_size=1/n_source)

    for train_index, test_index in sss.split(XX, yy):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_calib = XX[train_index], XX[test_index]
        y_train, y_calib = yy[train_index], yy[test_index]
        calib_pred, testPred = icp.ICPRegression(X_train, y_train, X_calib, y_calib, X_test,
                                                            method = "rf", nrTrees=10, returnPredictions = True)
        confScores = icp.computeConformityScores(calib_pred, y_calib)

        for i in range(len(sigLevels)):
            intervals[i, sourceIndex, :, :] = icp.computeInterval(confScores, testPred, sigLevels[i])

        #eff, errRate = pValues2PerfMetrics(intervals[sourceIndex], y_test)
        sourceIndex = sourceIndex+1
        #print(eff, errRate)


    for i in range(len(sigLevels)):
        #combined_intervals = combine_intervals(intervals[i])
        combined_intervals =  aggregateIntervals(intervals[i])
        eff, errorRate[i] = pValues2PerfMetrics(combined_intervals, y_test)

    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.plot(sigLevels, errorRate, color="green")
    plt.annotate("",
                 xy=(0, 0), xycoords='data',
                 xytext=(1, 1), textcoords='data',
                 arrowprops=dict(arrowstyle="-",
                                 connectionstyle="arc3,rad=0.",
                                 color='r')
                 )
    plt.show()





if __name__ == '__main__':
    X, y = data.load_superConduct_data()
    X, X_test, y, y_test \
            = train_test_split(X, y, test_size=0.3)
    #X, y = data.load_boston_data()
    synergyCP(X, y, n_source=3)
    ACP(X, y, n_source=3)

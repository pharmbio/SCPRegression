#############################################
# Author: Niharika Gauraha
# ICP: Inductive Conformal Prediction
#        for Regression using un-normalized
#        conformity measures
#############################################

from sklearn.ensemble import RandomForestRegressor
import numpy as np
import sys
from sklearn.svm import SVR, LinearSVR
from sklearn.model_selection import GridSearchCV
from math import exp, ceil, pi
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.datasets import load_boston
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from perf_measure import pValues2PerfMetrics
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error


scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Compute normalized conformity scores
def computeConformityScores(pred, y):
    res = abs(y - pred)
    return res


# Compute confidence intervals
def computeInterval(confScore, testPred, epsilon = 0.1):
    if confScore is None:
        sys.exit("\n NULL model \n")
    confScore = np.sort(confScore)
    nrTestCases  = len(testPred)
    intervals = np.zeros((nrTestCases,  2))

    for k in range(0, nrTestCases):
        # Compute threshold for split conformal, at level alpha.
        n = len(confScore)

        if (ceil((n) * epsilon) <= 1):
            q = np.inf
        else:
            q= (confScore[ceil((n) * (1 - epsilon))])

        intervals[k, 0] = testPred[k] - q
        intervals[k, 1] = testPred[k] + q

    return intervals


# fit SVR
def fit_SVR(X_train, y_train, X_calib, y_calib, testData):
    #grid_param = [{'kernel': ['rbf'], 'gamma': [.1,1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    grid_param = [{'kernel': ['rbf'], 'gamma': [1, .1, 1e-2, 1e-3],
                         'C': [1, 10, 100, 1000]}]

    clf = GridSearchCV(SVR(epsilon=0.01), grid_param, cv=5, scoring=scorer)
    clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    calibPred = clf.predict(X_calib)
    testPred = clf.predict(testData)

    return train_predict, calibPred, testPred



def fit_linearSVR(X_train, y_train, X_calib, y_calib, testData):
    #C_param = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16, 32]
    #C_param = [1 / 2, 1, 2, 4, 8, 16, 32]
    #grid_param = [{'C': C_param}]
    grid_param = [{'C': [1, 10, 100]}]

    #clf = GridSearchCV(LinearSVR(), grid_param, cv=5)
    clf = GridSearchCV(SVR(epsilon=0.01, kernel='linear'), grid_param, cv=5, scoring=scorer)
    clf.fit(X_train, y_train)
    C = clf.best_params_['C']
    print(C)
    train_predict = clf.predict(X_train)
    calibPred = clf.predict(X_calib)
    testPred = clf.predict(testData)

    return train_predict, calibPred, testPred



def fit_sigmoidSVR(X_train, y_train, X_calib, y_calib, testData):
    #grid_param = [{'gamma': [1e-2, 1e-3, 1e-4, 1e-5], 'C': [1, 10, 100, 1000]}]
    grid_param = {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                  'C': [1, 10, 100, 1000]}
     #'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
    clf = GridSearchCV(SVR(epsilon=.01, kernel='sigmoid'),
                       grid_param, cv=5, scoring=scorer)
    clf.fit(X_train, y_train)
    calibPred = clf.predict(X_calib)
    testPred = clf.predict(testData)
    train_predict = clf.predict(X_train)
    return train_predict, calibPred, testPred



def fit_polySVR(X_train, y_train, X_calib, y_calib, testData):
    grid_param = [{'C': [1, 10, 100, 1000]}]
    clf = GridSearchCV(SVR(epsilon=.01, kernel='poly'),
                       grid_param, cv=5, scoring=scorer)
    clf.fit(X_train, y_train)
    calibPred = clf.predict(X_calib)
    testPred = clf.predict(testData)
    train_predict = clf.predict(X_train)
    return train_predict, calibPred, testPred

# INK -spline of  order  zero
def ink_spline0_kernel(x1, x2):
    K = np.zeros((len(x1), len(x2)))
    for i in range(len(x1)):
        temp1 = x1[i, :]
        for j in range(len(x2)):
            temp2 = x2[j, :]
            temp = np.minimum(temp1, temp2)
            K[i, j] = np.product(temp)
    #return np.sum(temp)
    return K


# INK -spline of  order  one
def ink_spline1_kernel(x1, x2):
    K = np.zeros((len(x1), len(x2)))
    for i in range(len(x1)):
        temp1 = x1[i, :]
        for j in range(len(x2)):
            temp2 = x2[j, :]
            temp = (1/3) * (np.minimum(temp1, temp2))**3 + \
                   (1/2)*(np.minimum(temp1, temp2))**2 * (abs(temp1 - temp2))
            K[i, j] = np.product(temp)
    #return np.sum(temp)
    return K


def fit_splineSVR(X_train, y_train, X_calib, y_calib, testData):
    grid_param = [{'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}]
    clf = GridSearchCV(SVR(epsilon=.01, kernel=ink_spline1_kernel),
                       grid_param, cv=5, scoring=scorer)
    clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    calibPred = clf.predict(X_calib)
    testPred = clf.predict(testData)

    return train_predict, calibPred, testPred


def fit_RF(X_train, y_train, X_calib, y_calib, testData, nrTrees=10):
    clf = RandomForestRegressor(n_estimators=nrTrees, random_state=3)
    clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    calibPred = clf.predict(X_calib)
    testPred = clf.predict(testData)

    return train_predict, calibPred, testPred



def ICPRegression(X_train, y_train, X_calib, y_calib, X_test, method="rf",
                  returnPredictions = False, nrTrees=100):
    if (X_train is None) or (X_calib is None):
        sys.exit("\n 'training set' and 'calibration set' are required as input\n")

    if method =="linear_svr":
        print("Linear SVR")
        fit_function = fit_linearSVR
    elif method == "svr":
        print("SVR")
        fit_function = fit_SVR
    elif method == "sigmoid_svr":
        print("Sigmoid SVR")
        fit_function = fit_sigmoidSVR
    elif method == "poly_svr":
        print("Polynomial SVR")
        fit_function = fit_polySVR
    elif method == "spline_svr":
        print("Spline SVR")
        fit_function = fit_splineSVR
    else:
        print("RF")
        fit_function = fit_RF


    epsilon = .1
    train_predict, calib_pred, testPred = fit_function(X_train, y_train, X_calib, y_calib, X_test)

    confScores = computeConformityScores(calib_pred, y_calib)
    intervals = computeInterval(np.sort(confScores), testPred, epsilon)

    if returnPredictions:
        #return confScores, testPred
        return calib_pred, testPred

    return intervals, testPred


if __name__ == '__main__':
    def test_linear_SVR():
        n = 200
        p = 500
        s = 10
        x = np.random.normal(0,1, (n,p))
        beta = np.zeros(p)
        beta[0:s] = np.random.normal(0,1,s)
        y = np.matmul(x,beta) + np.random.normal(0,1,n)

        # Generate some example test data
        n0 = 1000
        x0 = np.random.normal(0,1, (n0,p))
        y0 = np.matmul(x0,beta)+ np.random.normal(0,1,n0)

        calib_pred, testPred   = fit_linearSVR(x[0:100],y[0:100],x[100:], y[100:], x0)
        confScores = computeConformityScores(calib_pred, y[100:])
        intervals = computeInterval(confScores, testPred, .05)
        # plot the shaded range of the confidence intervals
        plt.fill_between(range(n0), intervals[:,0], intervals[:,1],
                         color="blue", alpha=.5)
        # plot the mean on top
        plt.plot(x0[:,1], y0)
        plt.show()


    def test_rbf_SVR():
        n = 1000
        x = np.random.uniform(0, 2*pi, n)
        y = np.sin(x) + x*pi/30*np.random.normal(0,1, n)
        x = x.reshape(-1,1)

        # Generate some example test data
        n0 = 1000
        x0 = np.random.uniform(0, 2*pi, n0)
        y0 = np.sin(x0) + x0*pi/30*np.random.normal(0,1, n0)
        x0 = x0.reshape(-1,1)

        intervals, testPred = ICPRegression(x[0:100],y[0:100],x[100:],y[100:], x0,
                                            method='svr')
        mean_width = np.mean(abs(intervals[:, 0] - intervals[:, 1]))
        print(mean_width)
        intervals, testPred = ICPRegression(x[0:100], y[0:100], x[100:], y[100:], x0,
                                            method='lin_svr')
        mean_width = np.mean(abs(intervals[:, 0] - intervals[:, 1]))
        print(mean_width)
        #confScores = computeConformityScores(calib_pred, y[100:])
        #intervals = computeInterval(confScores, testPred, .1)
        # plot the shaded range of the confidence intervals
        # plot the mean on top

        #plt.fill_between(x0[:,0], intervals[:,1], intervals[:,0],
        #                 color="blue")
        eff, errRate = pValues2PerfMetrics(intervals, y0)
        #print(eff, errRate)

        plt.scatter(x0, y0)
        plt.scatter(x0, testPred, marker=".")
        plt.scatter(x0, intervals[:,0], marker=".", color="blue" )
        plt.scatter(x0, intervals[:,1], marker=".", color="blue")

        plt.show()


    def test_RF():
        n = 1000
        x = np.random.uniform(0, 2*pi, n)
        y = np.sin(x) + x*pi/30*np.random.normal(0,1, n)
        x = x.reshape(-1,1)

        # Generate some example test data
        n0 = 1000
        x0 = np.random.uniform(0, 2*pi, n0)
        y0 = np.sin(x0) + x0*pi/30*np.random.normal(0,1, n0)
        x0 = x0.reshape(-1,1)

        calib_pred, testPred  = fit_RF(x[0:100],y[0:100],x[100:],y[100:], x0)
        confScores = computeConformityScores(calib_pred, y[100:])
        intervals = computeInterval(confScores, testPred, .1)
        # plot the shaded range of the confidence intervals
        # plot the mean on top

        #plt.fill_between(x0[:,0], intervals[:,1], intervals[:,0],
        #                 color="blue")
        plt.scatter(x0, y0)
        plt.scatter(x0, testPred, marker=".")
        plt.scatter(x0, intervals[:,0], marker=".", color="blue" )
        plt.scatter(x0, intervals[:,1], marker=".", color="blue")

        plt.show()


    test_rbf_SVR()
    #test_RR()
    #test_boston_data()
    #test_RF()
#############################################
# Author: Niharika Gauraha
# ICP: Inductive Conformal Prediction
#        for Regression using un-normalized
#        conformity measures
#############################################

import numpy as np
import sys
from sklearn.svm import SVR, LinearSVR
from sklearn.model_selection import GridSearchCV
from math import exp, ceil, pi
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error


scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Compute normalized conformity scores
def computeConformityScores(pred, y):
    res = np.abs(y - pred)
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

    clf = GridSearchCV(SVR(epsilon=0.01), grid_param, cv=5, scoring=scorer, n_jobs=-1)
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

    #clf = GridSearchCV(SVR(epsilon=0.01, kernel='linear'), grid_param, cv=5, scoring=scorer)
    clf = GridSearchCV(SVR(epsilon=0.01, kernel='linear'), grid_param, cv=5, scoring=scorer, n_jobs=-1)
    clf.fit(X_train, y_train)
    C = clf.best_params_['C']
    print(C)
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
    else:
        print("RF")
        fit_function = fit_RF


    epsilon = .1
    train_predict, calib_pred, testPred = fit_function(X_train, y_train, X_calib, y_calib, X_test)

    confScores = computeConformityScores(calib_pred, y_calib)
    intervals = computeInterval(np.sort(confScores), testPred, epsilon)

    if returnPredictions:
        return calib_pred, testPred

    return intervals, testPred


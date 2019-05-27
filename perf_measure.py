"""
 Author: Niharika gauraha
 Computes various performance measures
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import math


# Computes efficiency of a conformal predictor, average width of the intervals
def Efficiency(intervals):
    nrTestCases = len(intervals)

    mean_width = np.mean(abs(intervals[:,0]-intervals[:,1]))

    return mean_width


def ErrorRate(intervals, testLabels):
    if (intervals is None) or (testLabels is None):
        sys.exit("\n NULL values for input parameters \n")

    nrTestCases = len(testLabels)

    err = 0

    for j in range(0, nrTestCases):
        if (intervals[j, 0] > testLabels[j]) or \
                (intervals[j, 1] < testLabels[j]):
            err = err + 1

    err = err / nrTestCases

    return err




#Function: pValues2PerfMetrics
#Desc: Computes performance measures: efficiency and validity of the conformal predictors
def pValues2PerfMetrics(intervals, testLabels):
  eff = Efficiency(intervals)
  errRate = ErrorRate(intervals, testLabels)
  return eff, errRate


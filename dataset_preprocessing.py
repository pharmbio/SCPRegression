"""
 Author: Niharika gauraha
 Synergy Conformal Prediction Using Random Forest Classifier
"""

import csv
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import random

def load_boston_data():
    boston = load_boston()
    X = boston.data
    y = boston.target

    return X, y


def load_energy_data():
    data = []
    # Read the training data
    file = open('data/ENB2012_data.csv')
    reader = csv.reader(file)

    for row in reader:
        data.append(row)
    file.close()

    X = np.array([x[:-2] for x in data]).astype(np.float)
    # There are two response variable heating load/cooling load
    # response variable is the last
    y = np.array([x[-1] for x in data]).astype(np.float)
    del data # free up the memory

    return X, y


def load_concrete_data():
    data = []
    # Read the training data
    file = open('data/Concrete_Data.csv')
    reader = csv.reader(file)

    for row in reader:
        data.append(row)
    file.close()

    X = np.array([x[:-1] for x in data]).astype(np.float)
    # response variable is the last one: Concrete compressive strength
    y = np.array([x[-1] for x in data]).astype(np.float)
    del data # free up the memory

    return X, y


def load_wine_data():
    data = []
    # Read the training data
    file = open('data/winequality-white.csv')
    reader = csv.reader(file)

    for row in reader:
        data.append(row)
    file.close()

    X = np.array([x[:-2] for x in data]).astype(np.float)
    # response variable is the second last one: alcohol
    # Output variable (based on sensory data)
    y = np.array([x[-2] for x in data]).astype(np.float)
    del data # free up the memory
    X_remove, X, y_remove, y = train_test_split(X, y, test_size=2000 / len(X), random_state=7)
    return X, y


def load_PD_data():
    data = []
    # Read the training data
    file = open('data/train_data.txt')
    reader = csv.reader(file)

    for row in reader:
        data.append(row)
    file.close()
    # skip the first colum, which is ID
    X = np.array([x[1:-2] for x in data]).astype(np.float)
    # response variable is the last one: TODO
    y = np.array([x[-1] for x in data]).astype(np.float)
    del data # free up the memory

    return X, y


def load_powerPlant_data():
    data = []
    # Read the training data
    file = open('data/Folds5x2_pp.csv')
    reader = csv.reader(file,  delimiter=',')

    for row in reader:
        data.append(row)
    file.close()

    X = np.array([x[:-1] for x in data]).astype(np.float)
    # response variable is the last one
    y = np.array([x[-1] for x in data]).astype(np.float)
    del data # free up the memory

    X_remove, X, y_remove, y = train_test_split(X, y, test_size=2000 / len(X), random_state=7)

    return X, y


def load_gridStability_data():
    data = []
    # Read the training data
    file = open('data/grid_stability.csv')
    reader = csv.reader(file)

    for row in reader:
        data.append(row)
    file.close()

    X = np.array([x[:-2] for x in data]).astype(np.float)
    # response variable is the second last one
    y = np.array([x[-2] for x in data]).astype(np.float)
    del data # free up the memory
    X_remove, X, y_remove, y = train_test_split(X, y, test_size=2000/len(X), random_state=7)

    return X, y


def load_CBM_data():
    data = []
    # Read the training data
    file = open('data/CBM_data.txt')
    reader = csv.reader(file, delimiter=',')
    #next(reader)
    for row in reader:
        data.append(row)
    file.close()

    X = np.array([x[1:-2] for x in data]).astype(np.float)
    # response variable is the last two columns
    y = np.array([x[-1] for x in data]).astype(np.float)
    del data # free up the memory

    X_remove, X, y_remove, y = train_test_split(X, y, test_size=2000 / len(X), random_state=7)
    return X, y


def load_superConduct_data():
    data = []
    # Read the training data
    file = open('data/super_conduct.csv')
    reader = csv.reader(file,  delimiter=',')
    next(reader)
    for row in reader:
        data.append(row)
    file.close()

    X = np.array([x[:-1] for x in data]).astype(np.float)
    # response variable is the last one
    y = np.array([x[-1] for x in data]).astype(np.float)
    del data # free up the memory
    #X, X_test, y, y_test = train_test_split(X, y, test_size=0.75, random_state=7)
    #X, X_test, y, y_test = train_test_split(X, y, test_size=0.5)
    X_remove, X, y_remove, y = train_test_split(X, y, test_size=2000 / len(X), random_state=7)
    return X, y


def load_game_data():
    data = []
    # Read the training data
    file = open('data/SkillCraft1_Dataset.csv')
    reader = csv.reader(file,  delimiter=',')
    next(reader)
    for row in reader:
        data.append(row)
    file.close()
    # skip first two columns , TODO: check for missing data
    X = np.array([x[2:] for x in data]).astype(np.float)
    # response variable is ordinal: the lead index, second colum
    y = np.array([x[1] for x in data]).astype(np.float)
    del data # free up the memory

    X_remove, X, y_remove, y = train_test_split(X, y, test_size=2000 / len(X), random_state=7)
    return X, y

# disjoint partition without resampling
def equal_partition(n_train, N=3, seed=3):
    # shuffle the index first
    random.seed(seed)
    randIndex = random.sample(list(range(n_train)), n_train)
    splitLen = n_train//N
    #print(n_train, N, splitLen)
    list_part = np.zeros((N, splitLen))
    list_part[0, :] = randIndex[0:splitLen]
    for i in range(1, N):
        startIndex = i*splitLen
        list_part[i, :] = randIndex[startIndex:(startIndex+splitLen)]

    list_part = list_part.astype(int)

    return list_part


if __name__ == '__main__':
    X, y = load_game_data()
    #X, y = load_boston_data()
    #X, y = load_energy_data()
    #X, y = load_concrete_data()
    #X, y = load_wine_data()
    #X, y = load_PD_data()
    #X, y = load_powerPlant_data()
    #X, y = load_gridStability_data()
    #X, y = load_CBM_data()
    #X, y = load_superConduct_data()
    print(X.shape)

    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    list_part = equal_partition(len(X_train), N=3)
    # list_part = equal_partition_overlap(len(X_train), N, size=2000)
    # print(list_part)
    for part_index in list_part:
        x_part = X_train[part_index]
        y_part = y_train[part_index]
        print(part_index)
    '''
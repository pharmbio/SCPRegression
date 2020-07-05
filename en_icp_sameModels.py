"""
 Author: Niharika gauraha
 Train ensemble ICP on partitions using same training algorithm
"""

import dataset_preprocessing as data
from collections import OrderedDict
from scp import synergyCP
import os
import time
import numpy as np
from ensemble_icp import ensembleICP


epsilon = 0.1
iteration = 10
nrSources = 3
methods = ['linear_svr'] * nrSources
path = 'json_enicp_sameModel_linear'

methods = ['svr'] * nrSources
path = 'json_enicp_sameModel_nonlinear'

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
# these are time taking, so train them separately
#dataset_names = ['Wine', 'GridStability', 'SuperConduct']
load_functions["Wine"] = data.load_wine_data
load_functions["GridStability"] = data.load_gridStability_data
load_functions["SuperConduct"] = data.load_superConduct_data

#dataset_names = ['Housing']

np.random.seed(123)

#s_time = time.time()
for dataset_name in dataset_names:
    X, y = load_functions[dataset_name]()

    if not os.path.exists(path):
        os.makedirs(path)

    file_name = path + "/" + dataset_name + ".json"

    print(dataset_name)
    ensembleICP(X, y, n_source=nrSources, methods=methods, path=file_name)

#print("time taken by linear model: ", time.time()-s_time)

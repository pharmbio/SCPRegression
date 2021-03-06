"""
 Author: Niharika gauraha
 Train SCP on partitions using same algorithm
"""

import dataset_preprocessing as data
from collections import OrderedDict
from scp import synergyCP
import os
import numpy as np


epsilon = 0.1
iteration = 10
nrSources = 3
methods = ['linear_svr', 'rf', 'svr']

dataset_names = ['Housing','PD', 'PowerPlant', 'Energy', 'Concrete', 'CBM', 'Game']
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

dataset_names = ['Housing']
np.random.seed(123)

for dataset_name in dataset_names:
    X, y = load_functions[dataset_name]()

    if not os.path.exists('json_diffModel'):
        os.makedirs('json_diffModel')

    file_name = "json_diffModel/" + dataset_name + ".json"

    print(dataset_name)
    synergyCP(X, y, n_source=nrSources, methods=methods, path=file_name)






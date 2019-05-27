from prettytable import PrettyTable
import numpy as np
import matplotlib.pyplot as plt
import  os
import json

# global variables:
dataset_names = ['Housing', 'Wine', 'PD', 'PowerPlant', 'Energy', 'Concrete',
                     'GridStability', 'SuperConduct', 'CBM', 'Game']

lin_icp = []
lin_list = []
rf_icp = []
rbf_icp = []
rbf_list = []
same_data_scp =[]

def plot_same_data():
    # plot for same data
    global lin_icp
    global rf_icp
    global rbf_icp
    global same_data_scp

    model = 'same_data'
    pt = PrettyTable()
    pt.field_names = ["Dataset", "SVR-ICP", "RF-ICP", "RBF-SVR-ICP", "SCP"]
    for dataset_name in dataset_names:
        file_name = "json_same_data/" + dataset_name + ".json"
        with open(file_name, 'r') as fh:
            results = json.loads(fh.read())

        source1 = results["source1"]
        source2 = results["source2"]
        source3 = results["source3"]
        SCP = results["SCP"]

        pt.add_row([dataset_name, round(np.median(source1), 3),
                    round(np.median(source2), 3),
                    round(np.median(source3), 3),
                    round(np.median(SCP), 3)])

        lin_list.append(source1)
        rbf_list.append(source3)

        lin_icp.append(round(np.median(source1),3))
        rf_icp.append(round(np.median(source2),3))
        rbf_icp.append(round(np.median(source3),3))
        same_data_scp.append(round(np.median(SCP), 3))

    #print(pt)



def plot_diff_models():
    # plot for same model non_linear
    model = 'diff_models'
    pt = PrettyTable()
    pt.field_names = ["Dataset", 'SVR-ICP$_p$', 'RF-ICP$_p$', 'RBF-SVR-ICP$_p$', 'SCP']
    for dataset_name in dataset_names:
        file_name = "json_diffModel/" + dataset_name + ".json"
        with open(file_name, 'r') as fh:
            results = json.loads(fh.read())

        source1 = results["source1"]
        source2 = results["source2"]
        source3 = results["source3"]

        SCP = results["SCP"]

        pt.add_row([dataset_name, round(np.median(source1), 3),
                    round(np.median(source2), 3),
                    round(np.median(source3), 3),
                    round(np.median(SCP), 3)])


        plt.xlim([0, 5])
        boxPlotLabels = ['SVR-ICP$_p$', 'RF-ICP$_p$','RBF-SVR-ICP$_p$', 'SCP']
        data = np.column_stack((source1, source2, source3, SCP))
        plt.boxplot(data, labels=boxPlotLabels, patch_artist=True)
        plt.ylabel("Efficiency")


        if not os.path.exists('plots/'+model):
            os.makedirs('plots/'+model)
        plt.savefig('plots/'+model+"/"+ dataset_name)
        plt.clf()

    print(pt)


#plot_diff_models()

def plot_same_model_linear():
    # plot for same model non_linear
    model = 'same_model_lin'
    pt = PrettyTable()

    indx_dataset = 0
    for dataset_name in dataset_names:
        file_name = "json_sameModel_linear/" + dataset_name + ".json"
        with open(file_name, 'r') as fh:
            results = json.loads(fh.read())

        source1 = results["source1"]
        source2 = results["source2"]
        source3 = results["source3"]

        min_index = np.argmin([round(np.median(source1), 3), round(np.median(source2), 3),
                               round(np.median(source3), 3)])
        if min_index==0:
            icp_p = source1
        elif min_index==1:
            icp_p = source2
        else:
            icp_p = source3

        SCP = results["SCP"]

        pt.field_names = ["Dataset", "ICP_p", "ICP", "SCP"]
        pt.add_row([dataset_name,
                    round(np.median(icp_p), 3),
                    lin_icp[indx_dataset],
                    round(np.median(SCP), 3)])


        plt.xlim([0, 4])
        boxPlotLabels = ["ICP_p", "ICP", "SCP"]
        data = np.column_stack((icp_p, lin_list[indx_dataset], SCP))
        plt.boxplot(data, labels=boxPlotLabels, patch_artist=True)
        plt.ylabel("Efficiency")
        indx_dataset += 1
    
        if not os.path.exists('plots/'+model):
            os.makedirs('plots/'+model)
        plt.savefig('plots/'+model+"/"+ dataset_name)
        plt.clf()

    print(pt)



def plot_same_model_nonlinear():
    # plot for same model non_linear
    model = 'same_model_nonlin'
    pt = PrettyTable()

    indx_dataset = 0
    for dataset_name in dataset_names:
        file_name = "json_sameModel_nonlinear/" + dataset_name + ".json"
        with open(file_name, 'r') as fh:
            results = json.loads(fh.read())

        source1 = results["source1"]
        source2 = results["source2"]
        source3 = results["source3"]

        min_index = np.argmin([round(np.median(source1), 3), round(np.median(source2), 3),
                               round(np.median(source3), 3)])
        if min_index == 0:
            icp_p = source1
        elif min_index == 1:
            icp_p = source2
        else:
            icp_p = source3

        SCP = results["SCP"]
        pt.field_names = ["Dataset", "ICP_p", "ICP", "SCP"]
        pt.add_row([dataset_name,
                    round(np.median(icp_p), 3),
                    rbf_icp[indx_dataset],
                    round(np.median(SCP), 3)])



        plt.xlim([0, 4])
        boxPlotLabels = ["ICP_p", "ICP", "SCP"]
        data = np.column_stack((icp_p, rbf_list[indx_dataset], SCP))
        plt.boxplot(data, labels=boxPlotLabels, patch_artist=True)
        plt.ylabel("Efficiency")
        indx_dataset += 1

        if not os.path.exists('plots/'+model):
            os.makedirs('plots/'+model)
        plt.savefig('plots/'+model+"/"+ dataset_name)
        plt.clf()

    print(pt)


def plot_same_data_acp():
    model = 'same_data'
    pt = PrettyTable()
    pt.field_names = ["Dataset", "SVR-ICP", "RF-ICP", "RBF-SVR-ICP", "SCP",
                      "SVR-CCP", "RF-CCP", "RBF-SVR-CCP"]
    indx_dataset = 0
    for dataset_name in dataset_names:
        # populate ICP and SCP data first
        file_name = "json_same_data/" + dataset_name + ".json"
        with open(file_name, 'r') as fh:
            results = json.loads(fh.read())

        source1 = results["source1"]
        source2 = results["source2"]
        source3 = results["source3"]
        SCP = results["SCP"]


        file_name = "json_acp/" + dataset_name + "linear_svr.json"
        with open(file_name, 'r') as fh:
            results = json.loads(fh.read())

        acp_lin = results


        file_name = "json_acp/" + dataset_name + "rf.json"
        with open(file_name, 'r') as fh:
            results = json.loads(fh.read())
        acp_rf = results

        file_name = "json_acp/" + dataset_name + "svr.json"
        with open(file_name, 'r') as fh:
            results = json.loads(fh.read())
        acp_rbf = results


        pt.add_row([dataset_name,
                    lin_icp[indx_dataset],
                    rf_icp[indx_dataset],
                    rbf_icp[indx_dataset],
                    same_data_scp[indx_dataset],
                    round(np.median(acp_lin), 3),
                    round(np.median(acp_rf), 3),
                    round(np.median(acp_rbf), 3)
                    ])
        indx_dataset += 1
        plt.figure(figsize=(8, 4))
        plt.xlim([0, 9])

        boxPlotLabels = ['SVR-ICP', 'RF-ICP','RBF-SVR-ICP', 'SCP',
                         'SVR-CCP', 'RF-CCP    ','RBF-SVR-CCP']
        data = np.column_stack((source1, source2, source3, SCP,
                                acp_lin, acp_rf, acp_rbf))
        plt.boxplot(data, labels=boxPlotLabels, patch_artist=True)
        plt.ylabel("Efficiency")


        if not os.path.exists('plots/'+model):
            os.makedirs('plots/'+model)
        plt.savefig('plots/'+model+"/"+ dataset_name)
        plt.clf()

    print(pt)

plot_same_data()
'''
print("diff models")
plot_diff_models()
print("same models linear")
plot_same_model_linear()
print("same models non-lin")
plot_same_model_nonlinear()
'''
print("same data")
plot_same_data_acp()

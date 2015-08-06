__author__ = 'Simon'

import os
import numpy as np
np.set_printoptions(threshold=np.nan)
import scipy as sp
import scipy.io as io
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
import sklearn.svm as svm
import utility as ut
import optunity
import optunity.metrics
import time
import math
import itertools
import matplotlib.pylab as plt
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import copy



BETA = 1
LAMBDA_L = 1
LAMBDA_D = 1
THR = 0.2
#C = 1
#EPSILON = 0.1

def second_phase(data, results):
    result_directory_name = 'results'
    result_filename = os.path.join(result_directory_name, 'secondPhase', 'results_main_secondPhase.txt')
    if not (os.path.exists(result_directory_name)):
        os.mkdir(result_directory_name)
    if not (os.path.exists(os.path.join(result_directory_name, 'secondPhase'))):
        os.mkdir(os.path.join(result_directory_name, 'secondPhase'))
    #scaler = StandardScaler()
    #Xdata = scaler.fit_transform(data.Xtarget)
    Xdata = copy.deepcopy(data.Xtarget)
    K = Xdata.dot(Xdata.T)
    tar_train_index = data.tar_train_index
    tar_test_index = data.tar_test_index
    all_test_dv = []
    mmd_values = []

    for r in range(data.nRound):
        #loading DVs and MMD values
        for s in range(len(data.Xsource)):
            dv_dir = os.path.join(result_directory_name, str(data.currentY), 'svm_fr', 'decision_values', data.domain_names[s])
            mmd_dir = os.path.join(result_directory_name, str(data.currentY), 'mmd_values_fr', data.domain_names[s])
            dv_file = os.path.join(dv_dir, "dv_round="+str(r)+".mat")
            if os.path.exists(dv_file):
                decision_values = io.loadmat(dv_file)['decision_values']
            else:
                print('You need to run the required baseline algorithms to obtain the decision values required by algorithm')
                return -1
            mmd_file = os.path.join(mmd_dir, 'mmd.mat')
            if os.path.exists(mmd_file):
                mmd_value = (io.loadmat(mmd_file))['mmd_value']
            else:
                print('please run the proper save_mmd first to prepare the mmd values required by this algorithm')
                return -1
            mmd_values.extend(mmd_value)
            all_test_dv.extend([decision_values])

        y = copy.deepcopy(data.ytarget)
        y[tar_test_index] = 0

        f_s = np.squeeze(np.array(all_test_dv))
        print("mmds", mmd_values)
        gamma_s = np.zeros((len(mmd_values)), dtype=float)
        mmd_aux = np.reshape(np.asarray(mmd_values).flatten(), (len(mmd_values), 1))
        mmd_aux = np.power(mmd_aux, 2).flatten()
        gamma_s = gamma_s + np.exp((-1*BETA)*mmd_aux)
        gamma_s = gamma_s/np.sum(gamma_s)
        print("gamma", gamma_s)
        theta1 = LAMBDA_L
        theta2 = LAMBDA_D

        dv = train_fast_dam(K, y, f_s, gamma_s, theta1, theta2, np.array([]), np.array([]))

        formatted_results = []
        for item in dv[tar_test_index]:
            formatted_results.extend(item)
        formatted_results = np.asarray(formatted_results).reshape((len(formatted_results),1))
        results = np.hstack((results, formatted_results))


        accuracy = ut.final_accuracy(np.squeeze(dv[tar_test_index]), np.squeeze(data.ytarget[tar_test_index]))
        print("Accuracy?!", accuracy, "\n")

        return results




def train_fast_dam(K, y, f_s, gamma_s, theta1, theta2, f_p, lambda_):
    n = K.shape[0]
    idx_l = np.where(y!=0)[0] #select the index array from the returned tuple
    idx_u = np.where(y==0)[0] #select the index array from the returned tuple
    II = np.ones(n)
    II[idx_u] = 1/np.sum(gamma_s)
    II[idx_l] = II[idx_l]/theta1
    II[idx_u] = II[idx_u]/theta2
    II = np.diag(II)


    if len(f_p)>0:
        Kp = (f_p.dot(f_p.T)/lambda_)
        hatK = K + II + Kp
    else:
        hatK = K + II


    haty = f_s.T.dot(gamma_s)
    haty = haty/np.sum(gamma_s)
    haty[idx_l] = copy.deepcopy(y[idx_l])

    #check the differences between y and haty. Is dividing by the sum of gammas ok? (should be 1 and therefore effectless)


    ind = np.where(np.abs(haty[idx_u]) < THR)
    v_ind = np.concatenate((idx_l, np.setdiff1d(idx_u, idx_u[ind])))

    #print("hatk", hatK[v_ind][:, v_ind])
    #print("haty", haty[v_ind])
    model = prepare_svm(hatK[v_ind][:, v_ind], haty[v_ind])
    model.support_ = v_ind[model.support_]

    if len(f_p)>0:
        Ktest = K+Kp
    else:
        Ktest = K
    #print("Testing data instances", Ktest[:, model.support_])


    aux1 = Ktest[:, model.support_]
    aux2 = aux1.dot(np.asarray(model.dual_coef_).flatten())
    dv = np.squeeze(np.array(aux2 + model.intercept_))
    return dv

def prepare_svm(K, y):
    param_grid={"C": np.logspace(-6, 4, 10),
                "epsilon": np.logspace(-6,4, 10)}
    regressor = svm.SVR(kernel="precomputed")
    svr = GridSearchCV(regressor, cv=3, param_grid=param_grid, n_jobs=4)
    svr.fit(K, y)
    svr_tuned_params = svr.best_params_
    print("tuned params", svr_tuned_params)
    tuned_svr = svm.SVR(kernel="precomputed", C=svr_tuned_params['C'], epsilon=svr_tuned_params['epsilon'])
    tuned_svr.fit(K, y)
    return tuned_svr






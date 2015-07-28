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
import time
import math
import itertools
import matplotlib.pylab as plt
from sklearn.grid_search import GridSearchCV



BETA = 100
LAMBDA_L = 1
LAMBDA_D1 = 1
LAMBDA_D2 = 1
THR = 0
C = 1
EPSILON = 0.1

def univerdam(data, results):
    result_directory_name = 'results'
    result_filename = os.path.join(result_directory_name, 'univerdam', 'results_main_univerdam.txt')
    if not (os.path.exists(result_directory_name)):
        os.mkdir(result_directory_name)
    if not (os.path.exists(os.path.join(result_directory_name, 'univerdam'))):
        os.mkdir(os.path.join(result_directory_name, 'univerdam'))
    X_data = []
    y_data = [[]]
    for indx in range(len(data.Xtarget)):
        X_data.append(data.Xtarget[indx])
    for ind in range(len(data.ytarget)):
        y_data[0].append(data.ytarget[ind])
    y_data[0] = np.asarray(y_data[0])
    domain_index = [[]]
    offset = len(data.ytarget)
    for j in range(len(data.ytarget)):
        domain_index[0].append(j) #todo used to be j+1
    for i in range(len(data.Xsource)):
        X_data.append(data.Xsource[i])
        y_data.append(data.ysource[i])
        domain_index.append([])
        for k in range(len(data.ysource[i])):
            domain_index[i+1].append(k+1+offset)
        offset += len(data.ysource)
    X = []
    X.extend(X_data[0])
    for j1 in range(len(X_data)-1):
        X.extend(X_data[j1+1])
    y = []
    y_data = np.asarray(y_data)


    for i1 in range(len(y_data)):
        for i2 in range(len(y_data[i1])):
            y.append(y_data[i1][i2])
    tar_index = domain_index[0]
    src_index = []
    for q in range(len(domain_index)-1):
        src_index.extend(domain_index[q+1])
    src_index = np.array(src_index).flatten()
    X_sparse = []
    for row in X_data:
        X_sparse.append(row)
    X_sparse = sp.sparse.vstack((row for row in X_sparse))
    K = X_sparse.dot(X_sparse.transpose())
    K = K.todense()

    for r in range(data.nRound):
        tar_train_index = np.squeeze(data.tar_train_index[r])
        tar_test_index = np.squeeze(data.tar_test_index[r])
        tar_train_index = np.asarray(tar_train_index)
        tar_test_index = np.asarray(tar_test_index)
        all_test_dv = []
        mmd_values = []
        # Definitions:
        # DV : Decision Values
        # MMD : Maximum Mean Discrepancy
        for s in range(len(data.Xsource)):
            # vérification du type de virtual_label_type (paramètre)
            # Pour l'instant on passe seulement du '_fr', mais le "if" ici prévoit
            # si on veut essayer d'autres types de classificateur (tel qu'essayé par
            # les auteurs)
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



        f_s = np.array(all_test_dv)
        gamma_s = np.zeros((len(mmd_values)), dtype=float)
        mmd_aux = np.reshape(np.asarray(mmd_values).flatten(), (len(mmd_values), 1))
        mmd_aux = np.power(mmd_aux, 2).flatten()
        gamma_s = gamma_s + np.exp((-1*BETA)*mmd_aux)
        gamma_s = gamma_s/np.sum(gamma_s)
        f_s = np.squeeze(f_s)
        virtual_label = (f_s.transpose().dot(gamma_s))

        tilde_y = np.array(y, dtype=float)
        tilde_y = tilde_y.reshape((len(tilde_y), 1))
        tilde_y[src_index] = 0
        tilde_y = tilde_y.flatten()

        tilde_y[tar_test_index] = virtual_label[tar_test_index]

        add_kernel = np.ones(len(tilde_y))
        #line 77 in matlab code
        add_kernel[src_index] = 1/LAMBDA_D2
        add_kernel[tar_train_index] = 1/LAMBDA_L
        add_kernel[tar_test_index] = 1/LAMBDA_D1/np.sum(gamma_s)
        ind = np.where(abs(virtual_label[tar_test_index]) < THR)
        v_ind = np.concatenate((src_index, tar_train_index, np.setdiff1d(tar_test_index, tar_test_index[ind])))
        Ymatrix = np.asarray(np.squeeze(tilde_y[v_ind]), dtype=float)
        ind_array = np.array([[i for i in range(len(v_ind))]])

        Xmatrix = K[v_ind][:, v_ind] + np.diag(add_kernel[v_ind])

        #prepare svr finds the best hyperparameters and returns the tuned and fitted SVR
        model = prepare_svr(Xmatrix, tilde_y[v_ind])

        model.support_ = csc_matrix(v_ind[model.support_])
        aux1 = np.squeeze(K[tar_index][:, model.support_.todense()])

        aux2 = aux1.dot(np.asarray(model.dual_coef_).flatten())
        dv = np.squeeze(np.array(aux2 - model.intercept_))
        formatted_results = []
        for item in dv[tar_test_index]:
            formatted_results.append([item])
        results = np.hstack((results, formatted_results))
        dec_values_to_test = dv[tar_test_index]
        for i in range(len(dec_values_to_test)):
            dec_values_to_test[i] = 1*np.sign(dec_values_to_test[i])


        accuracy = ut.final_accuracy(dv[tar_test_index], data.ytarget[tar_test_index])
        # todo here, should change accuracy vector based on the sign of the data in the vector (otherwise will probably always get zero)

        print("Accuracy?!", accuracy)

        return results


def prepare_svr(X, Y):
    param_grid={"C": [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                "epsilon": [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]}
    train_size = 1000
    svr = GridSearchCV(svm.SVR(kernel="precomputed"), cv=5, param_grid=param_grid)
    svr.fit(X, Y)
    svr_tuned_params = svr.best_params_
    tuned_svr = svm.SVR(kernel="precomputed", C=svr_tuned_params['C'], epsilon=svr_tuned_params['epsilon'])
    tuned_svr.fit(X, Y)
    return tuned_svr

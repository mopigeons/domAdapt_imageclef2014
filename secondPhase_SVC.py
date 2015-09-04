from pandas.core.common import _ABCGeneric

__author__ = 'Simon'

import os
import numpy as np
np.set_printoptions(threshold=np.nan)
import scipy.io as io
import baseClassification as bc
import copy



BETA = 1
LAMBDA_L = 1
LAMBDA_D = 1
THR = 0.2

def second_phase(data, results):
    result_directory_name = 'results'
    result_filename = os.path.join(result_directory_name, 'secondPhase', 'results_main_secondPhase.txt')
    if not (os.path.exists(result_directory_name)):
        os.mkdir(result_directory_name)
    if not (os.path.exists(os.path.join(result_directory_name, 'secondPhase'))):
        os.mkdir(os.path.join(result_directory_name, 'secondPhase'))

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

        dv = calc_decision_values(Xdata, y, f_s, gamma_s, theta1, theta2, np.array([]), np.array([]))

        formatted_results = []
        for item in dv[tar_test_index]:
            formatted_results.extend(item)
        formatted_results = np.asarray(formatted_results).reshape((len(formatted_results),1))
        results = np.hstack((results, formatted_results))

        print("\n")
        return results




def calc_decision_values(X, y, f_s, gamma_s, theta1, theta2, f_p, lambda_):
    idx_l = np.where(y!=0)[0] #select the index array from the returned tuple
    idx_u = np.where(y==0)[0] #select the index array from the returned tuple

    haty = f_s.T.dot(gamma_s)
    haty = haty/np.sum(gamma_s)
    haty[idx_l] = copy.deepcopy(y[idx_l])

    #check the differences between y and haty. Is dividing by the sum of gammas ok? (should be 1 and
    # therefore effectless)

    ind = np.where(np.abs(haty[idx_u]) < THR)
    v_ind = np.concatenate((idx_l, np.setdiff1d(idx_u, idx_u[ind])))
    #turn haty into 1s and -1s
    haty_class = np.zeros(len(haty))
    for i in range(len(haty)):
        if haty[i] >= 0:
            haty_class[i] = 1
        else:
            haty_class[i] = -1

    model = bc.prepare_svm(X[v_ind], haty_class[v_ind], True)

    dv = model.predict_proba(X)
    #take the prediction for class 1 (column 1)
    probabilities = []
    for i in range(len(dv)):
        probabilities.append(dv[i][1])
    print("DV", probabilities)
    return np.asarray(probabilities)


'''def prepare_svm(K, y):

    param_grid = {'C': np.logspace(-5, 4, 10), 'class_weight': [{1: 1}, {1: 11}, {1: 22}]}

    classifier = svm.SVC(kernel="precomputed")
    svc = GridSearchCV(classifier, cv=3, param_grid=param_grid, n_jobs=4)
    svc.fit(K, y)
    svc_tuned_params = svc.best_params_
    print("tuned params", svc_tuned_params)
    tuned_svc = svm.SVC(kernel="precomputed", C=svc_tuned_params['C'],
                        class_weight=svc_tuned_params['class_weight'], probability=True)
    tuned_svc.fit(K, y)
    return tuned_svc'''



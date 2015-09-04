__author__ = 'Simon'

import utility as ut
import os
import scipy.io as io
import numpy as np
import scipy as sp
from scipy.sparse import csc_matrix
import datetime
from sklearn.svm import SVC
import optunity
import optunity.metrics






def svm(data):
    #create results file if it doesn't already exist
    results_directory_name = "results"
    result_file = os.path.join(results_directory_name, str(data.currentY), "svm_fr", "result_main_svm_fr.txt")
    # crée le répertoire "results" et le sous-répertoire "svm_fr"
    if not (os.path.exists(results_directory_name)):
        os.mkdir(results_directory_name)
    if not (os.path.exists(os.path.join(results_directory_name, str(data.currentY)))):
        os.mkdir(os.path.join(results_directory_name, str(data.currentY)))
    if not (os.path.exists(os.path.join(results_directory_name, str(data.currentY), 'svm_fr'))):
        os.mkdir(os.path.join(results_directory_name, str(data.currentY), 'svm_fr'))
    ut.log_print(result_file,
                 '<==========  BEGIN @ ' + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") +  ' ============>\n')
    best_params = []
    for s in range(len(data.Xsource)):
        print("\n--- Domain:",s,"---\n")
        # création de répertoires results/svm_fr/decision_values/...
        dv_dir = os.path.join(results_directory_name, str(data.currentY), 'svm_fr', 'decision_values', data.domain_names[s])
        if not (os.path.exists(results_directory_name)):
            os.mkdir(results_directory_name)
        if not (os.path.exists(os.path.join(results_directory_name, str(data.currentY)))):
            os.mkdir(os.path.join(results_directory_name, str(data.currentY)))
        if not (os.path.exists(os.path.join(results_directory_name, str(data.currentY), 'svm_fr'))):
            os.mkdir(os.path.join(results_directory_name, str(data.currentY), 'svm_fr'))
        if not (os.path.exists(os.path.join(results_directory_name, str(data.currentY), 'svm_fr', 'decision_values'))):
            os.mkdir(os.path.join(results_directory_name, str(data.currentY), 'svm_fr', 'decision_values'))
        if not (os.path.exists(dv_dir)):
            os.mkdir(dv_dir)
        #for the s-th source domain...
        Xsource = data.Xsource[s]
        ysource = data.ysource[s]
        #glue together the labelled data from the source and target domains
        Xsparse = sp.sparse.vstack([data.Xtarget, csc_matrix(Xsource)])
        Xdata = Xsparse.todense()
        #get the associated labels for all training points
        y = np.concatenate((data.ytarget, ysource))
        src_index = [i + data.Xtarget.shape[0] for i in range(Xsource.shape[0])]
        tar_index = [i for i in range(data.Xtarget.shape[0])]

        for r in range(data.nRound):
            tar_train_index = data.tar_train_index[r]
            tar_test_index = data.tar_test_index[r]
            train_index = np.concatenate((src_index, tar_train_index))
            dv_file = os.path.join(dv_dir, "dv_round=" + str(r) + ".mat")
            if os.path.exists(dv_file):
                decision_values = io.loadmat(dv_file)['decision_values']
            else:
                Ymatrix = np.asarray(np.squeeze(y[train_index]), dtype=float)

                Xmatrix = np.asarray(Xdata[train_index])

                classifier = prepare_svm(Xmatrix, Ymatrix, False)
                classifier_params = classifier.get_params()
                best_params.append(classifier_params)

                decision_values = classifier.decision_function(Xdata[tar_index])
                decision_values = np.array(decision_values)

                training_predictions = classifier.predict(Xmatrix)
                print("Training Accuracy", ut.final_accuracy(training_predictions, y[train_index]))

                predictions = classifier.predict(Xdata[tar_index])
                print("Test Accuracy", ut.final_accuracy(predictions, y[tar_index]))
                io.savemat(dv_file, {'decision_values': decision_values})

                ut.save_mmd_fr(data, best_params, s)


def get_SVM_kernel_param(classifier):
    all_params = classifier.get_params()
    kernel_type = all_params['kernel']
    if kernel_type == 'rbf' or kernel_type == 'sigmoid':
        kernel_param = all_params['gamma']
    elif kernel_type == 'linear':
        kernel_param = 1
    elif kernel_type == 'poly':
        kernel_param = all_params['degree']
    return kernel_param


def prepare_svm(X, Y, prob_setting):
    '''
    Code inspired by http://optunity.readthedocs.org/en/latest/notebooks/notebooks/sklearn-svc.html#tune-svc-without-deciding-the-kernel-in-advance
    '''
    cv_decorator = optunity.cross_validated(x=X, y=Y, num_folds=10)
    space = {'kernel': {'linear': {'C': [0, 1000], 'class_weight_param': [1, 22]},
                        'rbf': {'logGamma': [-5, 1], 'C': [0, 1000], 'class_weight_param': [1, 22]},
                        'poly': {'degree': [2, 5], 'C': [0, 1000], 'coef0': [0, 100],
                                 'class_weight_param': [1, 22]}}}

    def train_model(x_train, y_train, kernel, C, logGamma, degree, coef0, classWeightParam):
        if kernel=='linear':
            model = SVC(kernel=kernel, C=C, class_weight={1: classWeightParam})
        elif kernel=='poly':
            model = SVC(kernel=kernel, C=C, degree=degree, coef0=coef0, class_weight={1: classWeightParam})
        elif kernel=='rbf':
            model = SVC(kernel=kernel, C=C, gamma=10 ** logGamma, class_weight={1: classWeightParam})
        else:
            raise ValueError("Unknown kernel function: %s" % kernel)
        model.fit(x_train, y_train)
        return model


    def svm_tuned_auroc(x_train, y_train, x_test, y_test, kernel='linear', C=0, logGamma=0, degree=0, coef0=0, class_weight_param=1):
        model = train_model(x_train, y_train, kernel, C, logGamma, degree, coef0, class_weight_param)
        decision_values = model.decision_function(x_test)
        return optunity.metrics.roc_auc(y_test, decision_values)

    svm_tuned_auroc = cv_decorator(svm_tuned_auroc)

    optimal_svm_pars, info, _ = optunity.maximize_structured(svm_tuned_auroc, space, num_evals=200)
    print("Optimal parameters:"+str(optimal_svm_pars))
    print("AUROC of tuned SVM: %1.3f" % info.optimum)
    classifier = build_svc(optimal_svm_pars, prob_setting)
    classifier.fit(X, Y)
    return classifier


def build_svc(optimal_parameters, prob_setting):
    optimal_kernel = optimal_parameters['kernel']
    if optimal_kernel == 'linear':
        model = SVC(kernel=optimal_parameters['kernel'], C=optimal_parameters['C'],
                    class_weight={1: optimal_parameters['class_weight_param']}, probability=prob_setting)
    elif optimal_kernel == 'poly':
        model = SVC(kernel=optimal_parameters['kernel'], C=optimal_parameters['C'], degree=optimal_parameters['degree'],
                    coef0=optimal_parameters['coef0'], class_weight={1: optimal_parameters['class_weight_param']},
                    probability=prob_setting)
    elif optimal_kernel == 'rbf':
        model = SVC(kernel=optimal_parameters['kernel'], C=optimal_parameters['C'],
                    gamma=10 ** optimal_parameters['logGamma'],
                    class_weight={1: optimal_parameters['class_weight_param']}, probability=prob_setting)
    else:
        raise ValueError("Unknown kernel function: %s" % optimal_kernel)
    return model

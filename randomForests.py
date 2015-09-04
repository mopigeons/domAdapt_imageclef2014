__author__ = 'Simon'

import utility_rf as ut
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from utility_rf import NUM_SOURCE_DATA_SETS, NUM_CLASSES
import copy


MIN_VOTES_PER_CLASS_THRESHOLD = 35
MIN_VL_PROB_THRESHOLD = 0.1

NUM_ESTIMATORS = 100
MAX_DEPTH = 15
MAX_FEATURES = "auto"
MIN_SAMPLES_SPLIT = 2



def base_rf(data, results):
    predictions = []
    probabilities = []
    for s in range(len(data.Xsource)):
        #for the s-th source domain...
        Xsource = data.Xsource[s]
        ysource = data.ysource[s]
        #create the xmatrix by concatenating training data 10 times, adding source domain data
        Xdata = copy.deepcopy(data.Xtarget[data.tar_train_index])
        Xtarget_aux = copy.deepcopy(data.Xtarget[data.tar_train_index])
        for i in range(10):
            Xdata = np.append(Xdata, Xtarget_aux, 1)

        # glue together the labelled data from the source and target domains
        #Xsparse = sp.sparse.vstack([data.Xtarget, csc_matrix(Xsource)])
        Xdata = np.append(np.squeeze(Xdata), Xsource, 0)
        # get the associated labels for all training points

        y = copy.deepcopy(data.ytarget[data.tar_train_index])
        y = np.squeeze(y)
        y_aux = copy.deepcopy(data.ytarget[data.tar_train_index])
        y_aux = np.squeeze(y_aux)
        for i in range(10):
            y = np.append(y, y_aux, 0)
        y = np.append(y, ysource, 0)

        if results.predictions[s] != 0:
            decision_values = results.predictions[s]
        else:
            Ymatrix = np.asarray(np.squeeze(y))
            Xmatrix = np.asarray(Xdata)

            forest = RandomForestClassifier(n_estimators=NUM_ESTIMATORS, n_jobs=-1, max_depth=MAX_DEPTH, max_features=MAX_FEATURES, min_samples_split=MIN_SAMPLES_SPLIT)
            forest = forest.fit(Xmatrix, Ymatrix)

            probabilities.append(forest.predict_proba(np.squeeze(data.Xtarget[data.tar_test_index])).tolist())

    results.predictions = np.array(probabilities)

    return results


def determine_relevance(results):
    # establish the median for each domain's class votes, as long as it has cast at least THRESHOLD votes in that class
    shape_predictions = np.asarray(results.predictions).shape
    vote_strength = np.zeros((shape_predictions[0], shape_predictions[2]))

    votes = []
    for i in range(NUM_SOURCE_DATA_SETS):
        votes.append([])
        for j in range(NUM_CLASSES):
            votes[i].append([])

    # for each domain
    for s in range(len(results.predictions)):
        # create a list of the strength of the votes for each class
        max_ind = np.argmax(results.predictions[s], 1)
        for i in range(len(max_ind)):
            #only include the vote if its better than the threshold
            if results.predictions[s][i][max_ind[i]] > MIN_VL_PROB_THRESHOLD:
                votes[s][max_ind[i]].append(results.predictions[s][i][max_ind[i]])

    for s in range(len(votes)):
        for c in range(len(votes[s])):
            if len(votes[s][c]) > MIN_VOTES_PER_CLASS_THRESHOLD:
                vote_strength[s][c] = np.median(votes[s][c])
            else:
                vote_strength[s][c] = 0

    for c in range(vote_strength.shape[1]):
        class_total = 0
        for s in range(vote_strength.shape[0]):
            class_total += vote_strength[s][c]
        for s in range(vote_strength.shape[0]):
            vote_strength[s][c] = vote_strength[s][c]/class_total

    results.relevance = vote_strength
    return results


def get_virtual_vote(results):
    max_ind = []
    for s in range(len(results.predictions)):
        max_ind.append(np.argmax(results.predictions[s], 1))
    max_ind = np.array(max_ind)
    virtual_labels = []
    for i in range(max_ind.shape[1]):
        class_votes = np.zeros(NUM_CLASSES)
        for s in range(max_ind.shape[0]):
            current_vote_strength = results.relevance[s][max_ind[s][i]] * results.predictions[s][i][max_ind[s][i]]
            if current_vote_strength > class_votes[max_ind[s][i]]:
                class_votes[max_ind[s][i]] += current_vote_strength
        virtual_labels.append(class_votes)


    virtual_labels = np.asarray(virtual_labels)
    vl_iftrue = np.argmax(virtual_labels, axis=1)

    legit_ind = []
    final_labels = np.zeros(len(virtual_labels))
    for i in range(len(virtual_labels)):
        if np.amax(virtual_labels[i], 0) >= MIN_VL_PROB_THRESHOLD:
            final_labels[i] = vl_iftrue[i]
            legit_ind.append(i)
        else:
            final_labels[i] = -1

    final_labels += 1
    return final_labels, legit_ind


def second_classifier_trainonall(data, virtual_labels, vl_ind):
    print("\nTARGET CLASSIFIER")
    Xdata = copy.deepcopy(data.Xtarget[data.tar_train_index])
    Xdata_aux = copy.deepcopy(data.Xtarget[data.tar_train_index])
    for i in range(10*4):
        Xdata = np.append(Xdata, Xdata_aux, 1)
    X_target_test = data.Xtarget[data.tar_test_index]
    X_target_test = np.squeeze(X_target_test)
    Xdata = np.squeeze(Xdata)
    Xdata = np.append(Xdata, X_target_test[vl_ind], 0)

    for s in range(len(data.Xsource)):
        Xdata = np.append(Xdata, data.Xsource[s], 0)
    labels = copy.deepcopy(data.ytarget[data.tar_train_index])
    labels_aux = copy.deepcopy(data.ytarget[data.tar_train_index])
    for i in range(10*4):
        labels = np.append(labels, labels_aux, 0)
    labels = np.append(labels, virtual_labels[vl_ind])
    for s in range(len(data.Xsource)):
        labels = np.append(labels, data.ysource[s], 0)

    forest = RandomForestClassifier(n_estimators=NUM_ESTIMATORS, n_jobs=-1, max_depth=MAX_DEPTH, max_features=MAX_FEATURES, min_samples_split=MIN_SAMPLES_SPLIT)
    forest = forest.fit(Xdata, labels)

    Xtest = np.squeeze(data.Xtarget[data.tar_test_index])
    ytest = np.squeeze(data.ytarget[data.tar_test_index])
    predictions = forest.predict(Xtest)

    print("Final accuracy: ", ut.accuracy_test(predictions, ytest))




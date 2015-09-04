__author__ = 'Simon'

import numpy as np
import os
import os.path
from sklearn.preprocessing import StandardScaler
from copy import deepcopy


NUM_SOURCE_DATA_SETS = 4
NUM_FILES_PER_SOURCE = 600
NUM_TAR_TRAIN_FILES = 60
NUM_TAR_TEST_FILES = 600
NUM_TAR_IND_FILES = 1
N_ROUND = 1
NUM_CLASSES = 12

class Data:
    def __init__(self):
        self.domain_names = []
        self.Xsource = []           # features du domaine source
        self.ysource = []           # labels du domaine source
        self.Xtarget = []           # features du domaine target
        self.ytarget = []           # labels du domaine target
        self.tar_train_index = []
        self.tar_test_index = []
        self.tar_background_index = []

class Results:
    def __init__(self):
        #equivalent des dv
        self.predictions = [0*i for i in range(NUM_SOURCE_DATA_SETS)]
        #equivalent des mmd
        #matrice de dimension NUM_CLASSES * NUM_SOURCE_DATA_SETS
        self.relevance = [0*i for i in range(NUM_SOURCE_DATA_SETS)]


# prepare les données à partir des données dans le répertoite data
def load_data_from_files():
    data = Data()
    data.domain_names.extend(['bing', 'caltech', 'pascal', 'imagenet'])
    for i in range(NUM_SOURCE_DATA_SETS):
        datafolder = os.path.join('data', "S0"+str(i)+"_"+data.domain_names[i])
        datalabels = os.path.join(datafolder, 'label.txt')
        domain_X_data = []
        for j in range(NUM_FILES_PER_SOURCE):
            datafile = os.path.join(datafolder, data.domain_names[i]+"_"+str(j+1)+".txt")
            domain_X_data.append(np.loadtxt(datafile))
        data.Xsource.append(domain_X_data)
        with open(datalabels) as file:
            labels = file.readlines()
        labels = convert_label_list(labels)
        data.ysource.append(labels)
    data.Xsource = np.asarray(data.Xsource)
    data.ysource = np.asarray(data.ysource)
    tar_train_folder = os.path.join('data', 'T00_sun_train')
    for i in range(NUM_TAR_TRAIN_FILES):
        tar_train_file = os.path.join(tar_train_folder, 'sun_'+str(i+1)+'.txt')
        data.Xtarget.append(np.loadtxt(tar_train_file))
    tar_train_label_file = os.path.join(tar_train_folder, 'label.txt')
    with open(tar_train_label_file) as tarfile:
        tarlabels = tarfile.readlines()
    tarlabels = convert_label_list(tarlabels)
    tar_test_folder = os.path.join('data', 'T01_sun_test')
    for i in range(NUM_TAR_TEST_FILES):
        tar_test_file = os.path.join(tar_test_folder, 'sun_'+str(i+1)+'.txt')
        data.Xtarget.append(np.loadtxt(tar_test_file))
    tar_test_label_file = os.path.join(tar_test_folder, 'label.txt')
    with open(tar_test_label_file) as tarfile:
        tartestlabels = tarfile.readlines()
    tartestlabels = convert_label_list(tartestlabels)
    data.ytarget.extend(tarlabels)
    data.ytarget.extend(tartestlabels)
    data.Xtarget = np.asarray(data.Xtarget)
    data.ytarget = np.squeeze(np.asarray(data.ytarget).transpose())
    scaler = StandardScaler()
    dataToScale = deepcopy(data.Xtarget)
    for i in range(len(data.Xsource)):
        dataToScale = np.concatenate((dataToScale, data.Xsource[i]))
    scaledData = scaler.fit_transform(dataToScale)
    data.Xtarget = scaledData[0:len(data.Xtarget)]
    for i in range(len(data.Xsource)):
        offset = len(data.Xtarget)
        for j in range(i):
            offset += len(data.Xsource[j])

        endpoint = offset+len(data.Xsource[i])
        data.Xsource[i] = scaledData[offset:endpoint]

    data.tar_train_index.append([i for i in range(NUM_TAR_TRAIN_FILES)])
    data.tar_train_index = np.asarray(data.tar_train_index)
    data.tar_test_index.append([i for i in range(NUM_TAR_TEST_FILES)])
    data.tar_test_index = np.asarray(data.tar_test_index) + NUM_TAR_TRAIN_FILES
    return data


#labels changed as follows as described on http://imageclef.org/2014/adaptation
#1 aeroplane
#2 bike
#3 bird
#4 boat
#5 bottle
#6 bus
#7 car
#8 dog
#9 horse
#10 monitor
#11 motorbike
#12 people
def convert_label_list(labels):
    convertedlabels = np.zeros(len(labels), dtype=int)
    for i in range(len(labels)):
        if " aeroplane" in labels[i]:
            convertedlabels[i] = 1
        elif " bike" in labels[i]:
            convertedlabels[i] = 2
        elif " bird" in labels[i]:
            convertedlabels[i] = 3
        elif " boat" in labels[i]:
            convertedlabels[i] = 4
        elif " bottle" in labels[i]:
            convertedlabels[i] = 5
        elif " bus" in labels[i]:
            convertedlabels[i] = 6
        elif " car" in labels[i]:
            convertedlabels[i] = 7
        elif " dog" in labels[i]:
            convertedlabels[i] = 8
        elif " horse" in labels[i]:
            convertedlabels[i] = 9
        elif " monitor" in labels[i]:
            convertedlabels[i] = 10
        elif " motorbike" in labels[i]:
            convertedlabels[i] = 11
        elif " people" in labels[i]:
            convertedlabels[i] = 12
        else:
            convertedlabels[i] = -1
    return convertedlabels


def accuracy_test(predictions, truths):
    num_correct = 0
    class_scores = np.zeros(NUM_CLASSES)
    total = len(predictions)
    for i in range(len(predictions)):
        if predictions[i] == truths[i]:
            num_correct += 1
            class_scores[predictions[i]-1] += 1
    print("Number correct:", num_correct, "out of ", total)
    print("Class scores:", class_scores)
    if total == 0:
        return 0
    else:
        return num_correct/total



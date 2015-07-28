import scipy.io as io
import numpy as np
import scipy as sp
import os
import os.path
from scipy.sparse import csc_matrix
from scipy.spatial.distance import pdist, squareform

NUM_SOURCE_DATA_SETS = 4
NUM_FILES_PER_SOURCE = 600
NUM_TAR_TRAIN_FILES = 60
NUM_TAR_TEST_FILES = 600
NUM_TAR_IND_FILES = 1
N_ROUND = 1

class Data:
    def __init__(self):
        self.domain_names = []
        self.Xsource = []           # features du domaine source
        self.ysource = []           # labels du domaine source
        self.Xtarget = []           # features du domaine target
        self.ytarget = []           # labels du domaine target
        self.nRound = 0             # nb de fois qu'on repete l'experience. un different sample de 20 elements etiqueté est choisi pour chaque round.
        self.currentY = 0               # pour le multiclass. quel y évalue-t-on dans ce passage.
        self.tar_train_index = []
        self.tar_test_index = []
        self.tar_background_index = []


def getLabelsFromFile(filename):
    file = open(filename, 'r')

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
def convert_label_list(labels, currentLabel):
    for i in range(len(labels)):
        if currentLabel == 1 and " aeroplane" in labels[i]:
            labels[i] = 1
        elif currentLabel == 2 and " bike" in labels[i]:
            labels[i] = 1
        elif currentLabel == 3 and " bird" in labels[i]:
            labels[i] = 1
        elif currentLabel == 4 and" boat" in labels[i]:
            labels[i] = 1
        elif currentLabel == 5 and " bottle" in labels[i]:
            labels[i] = 1
        elif currentLabel == 6 and " bus" in labels[i]:
            labels[i] = 1
        elif currentLabel == 7 and " car" in labels[i]:
            labels[i] = 1
        elif currentLabel == 8 and " dog" in labels[i]:
            labels[i] = 1
        elif currentLabel == 9 and " horse" in labels[i]:
            labels[i] = 1
        elif currentLabel == 10 and " monitor" in labels[i]:
            labels[i] = 1
        elif currentLabel == 11 and " motorbike" in labels[i]:
            labels[i] = 1
        elif currentLabel == 12 and " people" in labels[i]:
            labels[i] = 1
        else:
            labels[i] = -1
    return labels


# prepare les données à partir des données dans le répertoite data
def load_data_from_files(index):
    data = Data()
    data.currentY = index
    data.nRound = N_ROUND
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
        labels = convert_label_list(labels, data.currentY)
        data.ysource.append(labels)
        #data.ysource.append(np.loadtxt(datalabels, delimiter='\n'))
    data.Xsource = np.asarray(data.Xsource)
    data.ysource = np.asarray(data.ysource)
    tar_train_folder = os.path.join('data', 'T00_sun_train')
    for i in range(NUM_TAR_TRAIN_FILES):
        tar_train_file = os.path.join(tar_train_folder, 'sun_'+str(i+1)+'.txt')
        data.Xtarget.append(np.loadtxt(tar_train_file))
    tar_train_label_file = os.path.join(tar_train_folder, 'label.txt')
    with open(tar_train_label_file) as tarfile:
        tarlabels = tarfile.readlines()
    tarlabels = convert_label_list(tarlabels, data.currentY)

    tar_test_folder = os.path.join('data', 'T01_sun_test')
    for i in range(NUM_TAR_TEST_FILES):
        tar_test_file = os.path.join(tar_test_folder, 'sun_'+str(i+1)+'.txt')
        data.Xtarget.append(np.loadtxt(tar_test_file))
    tar_test_label_file = os.path.join(tar_test_folder, 'label.txt')
    with open(tar_test_label_file) as tarfile:
        tartestlabels = tarfile.readlines()
    tartestlabels = convert_label_list(tartestlabels, data.currentY)

    data.ytarget.extend(tarlabels)
    data.ytarget.extend(tartestlabels)
    data.Xtarget = np.asarray(data.Xtarget)
    data.ytarget = np.squeeze(np.asarray(data.ytarget).transpose())


    data.tar_train_index.append([i for i in range(NUM_TAR_TRAIN_FILES)])
    data.tar_train_index = np.asarray(data.tar_train_index)
    data.tar_test_index.append([i for i in range(NUM_TAR_TEST_FILES)])
    data.tar_test_index = np.asarray(data.tar_test_index) + NUM_TAR_TRAIN_FILES
    return data



def calc_kernel(kernel_type, kernel_param, X):
    if kernel_type == "linear":
        S = X.dot(X.T)
        K = S
    elif kernel_type == "poly":
        S = X.dot(X.T)
        K = np.power((S+1), kernel_param)
    elif kernel_type == "rbf":
        pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))
        K = sp.exp(-pairwise_sq_dists * kernel_param)
    else:
        print("error in calc_kernel : kernel_type unrecognized!")
    return K

def calc_ap(gt, desc):
    assert len(gt) == len(desc)
    #gt = np.asarray(gt).flatten()
    #desc = np.asarray(desc).flatten()
    desc *= -1
    ind = desc.argsort()
    dv = desc
    dv.sort()
    dv = (-1*dv)
    gt = gt[ind]
    pos_ind = np.where(gt > 0)  # tuple where first element is the array containing the elements where gt[i] > 0
    npos = len(pos_ind[0])
    if npos == 0:
        ap = 0
    else:
        npos_array = np.array(range(npos))+1
        pos_ind_array = np.array(pos_ind) + 1
        divarray = (npos_array/pos_ind_array)
        ap = np.average(divarray)
    return [ap]


def final_accuracy(predictions, truths):
    assert len(predictions) == len(truths)
    correct = 0
    total = 0
    for i in range(len(predictions)):
        total += 1
        if np.sign(predictions[i]) == truths[i]:
            correct += 1
    if total > 0:
        return (correct/total)
    else:
        return 0





def log_print(log_file, varargin):
    with open(log_file, "a") as file:
        file.write(varargin)


def save_mmd_fr(data, kernel_type, kernel_param):
    #MMD: "Maximum Mean Discrepancy"
    result_dir = 'results'
    for s in range(len(data.Xsource)):
        Xsource = data.Xsource[s]
        ysource = data.ysource[s]

        mmd_dir = os.path.join(result_dir, str(data.currentY), 'mmd_values_fr', data.domain_names[s])
        if not (os.path.exists(result_dir)):
            os.mkdir(result_dir)
        if not (os.path.exists(os.path.join(result_dir, str(data.currentY)))):
            os.mkdir(os.path.join(result_dir, str(data.currentY)))
        if not (os.path.exists(os.path.join(result_dir, str(data.currentY), 'mmd_values_fr'))):
            os.mkdir(os.path.join(result_dir, str(data.currentY), 'mmd_values_fr'))
        if not (os.path.exists(mmd_dir)):
            os.mkdir(mmd_dir)

        Xstacked = np.concatenate((data.Xtarget, Xsource), axis=0)

        src_index = [i + data.Xtarget.shape[0] for i in range(Xsource.shape[0])]
        #tar_index = np.array([i for i in range(data.Xtarget[0].shape[0])]).transpose()
        tar_index = np.array([i for i in range(data.Xtarget.shape[0])]).transpose()
        #ss = np.zeros((len(src_index)+len(tar_index), 1))
        ss = np.zeros(len(src_index)+len(tar_index))

        ss[src_index] = 1/len(src_index)
        ss[tar_index] = -1/len(tar_index)
        K = calc_kernel(kernel_type, kernel_param, Xstacked)
        #K[np.ix_(src_index, src_index)] = np.multiply(K[src_index][:, src_index], 2)
        #K[np.ix_(tar_index, tar_index)] = np.multiply(K[tar_index][:, tar_index], 2)
        mmd_file = os.path.join(mmd_dir, 'mmd.mat')
        if os.path.exists(mmd_file):
            mmd_value = io.loadmat(mmd_file)['mmd_value']
        else:
            #mmd_value = (ss.transpose().flatten().dot(K)).dot(ss.flatten())
            K = np.squeeze(K)
            mmdvalaux = ss.dot(K)
            mmd_value = mmdvalaux.dot(ss.transpose())
            io.savemat(mmd_file, {'mmd_value': mmd_value})


def cast_votes(results):
    for i in range(len(results)):
        print(i+1)
        print(results[i])




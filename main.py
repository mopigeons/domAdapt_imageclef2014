import utility as ut
import utility_rf as utrf
import baseClassification as bC
import randomForests as rf
#import secondPhase_FastDAM as sec
import secondPhase_SVC as sec
import numpy as np

NUM_CLASSES = 12
#NUM_CLASSES = 4
NUM_ITERATIONS = 1

def main_dam():
    print("******* BASE CLASSIFICATION *******")
    for i in range(NUM_CLASSES):
        index = i+1
        print("\n*** Label:", index, "***")
        data = ut.load_data_from_files(index)
        bC.svm(data)
    print("\n\n******* SECOND PHASE *******")
    results = np.zeros((ut.NUM_TAR_TEST_FILES, 1))
    for j in range(NUM_CLASSES):
        index = j+1
        data = ut.load_data_from_files(index)
        results = sec.second_phase(data, results)
    if results.shape[1] > 1:
        results = np.delete(results, 0, 1)
    ut.cast_votes(results)


def main_random_forest():
    for i in range(NUM_ITERATIONS):
        print("************ RANDOM FOREST METHOD:",i+1,"************")
        data = utrf.load_data_from_files()
        results = rf.base_rf(data, utrf.Results())
        results = rf.determine_relevance(results)
        print("\n--- Virtual labels ---")
        virtual_labels, indices = rf.get_virtual_vote(results)
        ytarget = np.squeeze(data.ytarget[data.tar_test_index])
        print("\nVirtual label accuracy:", utrf.accuracy_test(virtual_labels[indices], ytarget[indices]))
        rf.second_classifier_trainonall(data, virtual_labels, indices)


if __name__ == "__main__":
    #main_dam()
    main_random_forest()




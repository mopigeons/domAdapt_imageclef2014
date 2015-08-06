import utility as ut
import baseClassification as bC
import secondPhase as sec
import numpy as np

NUM_CLASSES = 12
#NUM_CLASSES = 4

def main():
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
    if (results.shape[1]>1):
        results = np.delete(results, 0, 1)
    ut.cast_votes(results)




if __name__ == "__main__":
    main()
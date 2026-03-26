import pandas as pd
import numpy as np
import time
import numpy as np
import umap
from Levenshtein import distance as lev_dist # Example distance metric

task_name_to_seq_column = {
    'SpliceAI': 'sequence',
}

# aligner = Align.PairwiseAligner()
# aligner.match_score = 0
# aligner.mismatch_score = -1
# aligner.open_gap_score = -0.5
# aligner.extend_gap_score = -0.25

def load_train_test_val_sequences(task_name):
    column = task_name_to_seq_column[task_name]
    train = pd.read_csv(f'data/{task_name}/train.csv')[column].to_list()
    test = pd.read_csv(f'data/{task_name}/test.csv')[column].to_list()
    val = pd.read_csv(f'data/{task_name}/val.csv')[column].to_list()
   
    return train, test, val

tasks_names = ['SpliceAI']
for task_name in tasks_names:
    train, test, val = load_train_test_val_sequences(task_name)

    timer = time.time()
    distasnce_matrix = np.zeros((len(train), len(test)), dtype=np.uint16)
    for trainidx, train_seq in enumerate(train):
        if trainidx % 100 == 0:
            elapsed_time = time.time() - timer
            remaining_time = elapsed_time / (trainidx + 1) * (len(train) - trainidx - 1) / 60
            print(f'Elapsed time: {elapsed_time/60:.2f}m, Remaining time: {remaining_time:.2f}m', end='\r')
        for testidx, test_seq in enumerate(test):
            dist = lev_dist(train_seq, test_seq)
            distasnce_matrix[trainidx, testidx] = dist

    np.save(f'data/{task_name}/train-test-distances.npy', distasnce_matrix)
    



    

    

# 'clusters' is a pandas DataFrame mapping Sequence ID to Cluster ID
# Use this to ensure sequences from the same Cluster ID aren't in both Train and Test.
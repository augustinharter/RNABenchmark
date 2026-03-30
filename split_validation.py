#%%
import numpy as np
import pandas as pd
import time
import umap
from Levenshtein import distance as lev_dist # Example distance metric
import subprocess
import os
#%%
task_name_to_seq_column = {
    'SpliceAI': 'sequence',
}
tasks_names = ['SpliceAI']

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

#%%
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
    
#%%
for task_name in tasks_names:
    distasnce_matrix = np.load(f'data/{task_name}/train-test-distances.npy')
    print(f'{task_name} distance matrix shape: {distasnce_matrix.shape}')

#%% write one fasta files for train, val and test sequences
for task_name in tasks_names:
    train, test, val = load_train_test_val_sequences(task_name)
    combined = train + test + val
    with open(f'data/{task_name}/combined_sequences.fasta', 'w') as f:
        for idx, seq in enumerate(combined):
            f.write(f'>seq_{idx}\n{seq}\n')

    
#%% mmseq2 clustering
def run_mmseqs_clustering(input_fasta, output_prefix, min_id=0.3, coverage=0):
    tmp_dir = "tmp_mmseqs"
    
    # Ensure tmp directory exists
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # Construct the command
    # --search-type 3 is mandatory for RNA/DNA
    cmd = [
        "mmseqs", "easy-cluster",
        input_fasta,
        output_prefix,
        tmp_dir,
        "--min-seq-id", str(min_id),
        "-c", str(coverage),
        "-v", "3",  # Verbosity level
        "--dbtype", "2",
        "--gpu", "1",  # Use GPU if available
        "-s", "10",  # Sensitivity level (adjust as needed)
        "-k", "5",
    ]

    try:
        print(f"Running MMseqs2 on {input_fasta}...")
        subprocess.run(cmd, check=True)
        print("Clustering complete.")
        
        # Load the results into Pandas
        cluster_file = f"{output_prefix}_cluster.tsv"
        df = pd.read_csv(cluster_file, sep='\t', names=['representative', 'member'])
        return df

    except subprocess.CalledProcessError as e:
        print(f"Error running MMseqs2: {e}")
        return None

#%%
for task_name in tasks_names:
    df = run_mmseqs_clustering(f'data/{task_name}/combined_sequences.fasta', f'data/{task_name}/rna_clusters', min_id=0.3, coverage=0.0)
    print(df)
# 'clusters' is a pandas DataFrame mapping Sequence ID to Cluster ID
# Use this to ensure sequences from the same Cluster ID aren't in both Train and Test.
# %%

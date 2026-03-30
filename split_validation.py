#%%
import numpy as np
import pandas as pd
import time
import umap
from Levenshtein import distance as lev_dist # Example distance metric
from Levenshtein import hamming as ham_dist
import subprocess
import os
#%%
task_name_to_seq_column = {
    'SpliceAI': 'sequence',
    'Modification': 'sequence',
}
tasks_names = ['Modification'] 
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

#%%  DISTANCE MATRIX
for task_name in tasks_names:
    train, test, val = load_train_test_val_sequences(task_name)
    combined = train + test + val

    timer = time.time()
    distance_matrix = np.zeros((len(combined), len(combined)), dtype=np.uint8)
    for idx, seq in enumerate(combined):
        if idx % 10 == 0:
            elapsed_time = time.time() - timer
            remaining_time = elapsed_time / (idx + 1) * (len(combined) - idx - 1) / 60
            print(f'Progress: {(idx+1)/len(combined):.2%}  elapsed time: {elapsed_time/60:.2f}m, Remaining time: {remaining_time:.2f}m', end='\r')
        for otheridx, otherseq in enumerate(combined[idx+1:], start=idx+1):
            dist = ham_dist(seq, otherseq)
            distance_matrix[idx, otheridx] = dist

    np.save(f'data/{task_name}/ham-distances.npy', distance_matrix)
    
#%% LOAD DISTANCE MATRIX
for task_name in ['SpliceAI']:
    distance_matrix = np.load(f'data/{task_name}/ham-distances.npy')
    print(f'{task_name} distance matrix shape: {distance_matrix.shape}')

#%% Pick random pairs
for task_name in tasks_names:
    train, test, val = load_train_test_val_sequences(task_name)
    x, y = np.random.choice(len(train), 10000, replace=False), np.random.choice(len(test), 10000, replace=False)
    sampled_distances = distance_matrix[x, y]
    print(f'{task_name} sampled distances: {sampled_distances[:10]}')
    # plot histogram of sampled distances
    from matplotlib import pyplot as plt
    plt.hist(sampled_distances, bins=50)
    plt.title(f'{task_name} Levenshtein Distances')
    plt.show()
    plt.savefig(f'{task_name}_distance_histogram.png')
    
#%% UMAP Visualization
for task_name in tasks_names:
    distance_matrix = np.load(f'data/{task_name}/train-test-distances.npy')
    # UMAP expects a square distance matrix, so we can symmetrize it by taking the average of the train-test and test-train distances
    symmetric_distance_matrix = (distance_matrix + distance_matrix.T) / 2
    reducer = umap.UMAP(metric='precomputed', random_state=42)
    embedding = reducer.fit_transform(symmetric_distance_matrix)
    
    plt.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.5)
    plt.title(f'{task_name} UMAP Embedding of Train-Test Distances')
    plt.show()

    
#%% FASTA FILE write one fasta files for train, val and test sequences
for task_name in tasks_names:
    train, test, val = load_train_test_val_sequences(task_name)
    combined = train + test + val
    with open(f'data/{task_name}/combined_sequences.fasta', 'w') as f:
        for idx, seq in enumerate(combined):
            f.write(f'>seq_{idx}\n{seq}\n')
            
#%% mmseq2 clustering
def run_mmseqs_clustering(input_fasta, output_prefix, min_id=0.3, coverage=0.8):
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
        "--dbtype", "0",
        "--gpu", "1",  # Use GPU if available
        #"-s", "10",  # Sensitivity level (adjust as needed)
        #"-k", "5",
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
    df = run_mmseqs_clustering(f'data/{task_name}/combined_sequences.fasta', f'data/{task_name}/rna_clusters', min_id=0.3, coverage=0.8)
    print(df)
    print(f'{task_name} clustering results: {len(set(df["representative"]))} clusters found.')
# 'clusters' is a pandas DataFrame mapping Sequence ID to Cluster ID
# Use this to ensure sequences from the same Cluster ID aren't in both Train and Test.
# %%
run_mmseqs_clustering('test/DB.fasta', 'test')
# %%

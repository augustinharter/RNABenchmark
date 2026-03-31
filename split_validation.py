#%%
import sys
import numpy as np
import pandas as pd
import time
import umap
from Levenshtein import distance as lev_dist # Example distance metric
from Levenshtein import hamming as ham_dist
import subprocess
import os
from multiprocessing import Pool
from matplotlib import pyplot as plt
#%%
task_name_to_seq_column = {
    'SpliceAI': 'sequence',
    'Modification': 'sequence',
}
tasks_names = ['Modification'] 
tasks_names = sys.argv[1:] if len(sys.argv) > 1 else tasks_names

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
def compute_partial_distance_matrix(sequences, distance_func, start_idx=0, end_idx=None, process_id=0, max_processes=1):
    ysize = end_idx - start_idx
    xsize = len(sequences) - start_idx
    size = xsize * ysize - (ysize * (ysize + 1)) // 2  # Total number of pairwise comparisons in this block
    distance_matrix = np.zeros((ysize, xsize), dtype=np.uint8)
    counter = 0
    for idx, seq in enumerate(sequences[start_idx:end_idx], start=start_idx):
        for otheridx, otherseq in enumerate(sequences[idx+1:], start=idx+1):
            dist = distance_func(seq, otherseq)
            distance_matrix[idx-start_idx, otheridx-start_idx] = dist
            counter += 1
            if counter % 1e5 == 0:  # Print progress every 100k comparisons for the first process
                timestamp = int(time.time())
                if timestamp % max_processes == process_id:  # Print every minute
                    print(f'Process {process_id}: {(counter/size):.2%}', end='\r')

    return distance_matrix

#%% RUN DISTANCE MATRIX ON MULTI CPUS
if __name__ == "__main__":
    for task_name in tasks_names:
        distance_matrix = None
        print(f'Processing task: {task_name}')
        train, test, val = load_train_test_val_sequences(task_name)
        combined = train + test + val
        #combined = combined[:10000]  # Limit to 1k sequences for testing
        distance_matrix = np.zeros((len(combined), len(combined)), dtype=np.uint8)
        split_starts = np.arange(0, 1, 1/os.cpu_count())**2
        split_ends = np.arange(1/os.cpu_count(), 1+1/os.cpu_count(), 1/os.cpu_count())**2
        start_idxs = (split_starts * len(combined)).astype(int)
        end_idxs = (split_ends * len(combined)).astype(int)
        
        process_count = os.cpu_count()
        with Pool(processes=process_count) as pool:
            results = pool.starmap(compute_partial_distance_matrix, [(combined, lev_dist, start_idxs[idx], end_idxs[idx], idx, process_count) for idx in range(len(start_idxs))])
            print()
            for idx, partial in enumerate(results):
                distance_matrix[start_idxs[idx]:end_idxs[idx], start_idxs[idx]:] = partial
        
        distance_matrix = distance_matrix + distance_matrix.T  # Symmetrize the matrix
        np.save(f'data/{task_name}/distances.npy', distance_matrix)
   
    
        #%% LOAD DISTANCE MATRIX
        #distance_matrix = np.load(f'data/{task_name}/distances.npy')
        print(f'{task_name} distance matrix shape: {distance_matrix.shape}')
        size = distance_matrix.shape[0]
        #combined_len = 10000
        x, y = np.random.choice(size, 10000, replace=False), np.random.choice(size, 10000, replace=False)
        sampled_distances = distance_matrix[x, y]
        print(f'{task_name} sampled distances: {sampled_distances[:10]}')
        # plot histogram of sampled distances
        plt.hist(sampled_distances, bins=100)
        plt.title(f'{task_name} Levenshtein Distances 10k Samples')
        plt.savefig(f'data/{task_name}/distance_histogram.png')
        
        #%% UMAP Visualization
        distance_matrix = np.load(f'data/{task_name}/distances.npy')
        print(distance_matrix.shape)
        subsample_size = 10000
        if distance_matrix.shape[0] > subsample_size:
            indices = np.random.choice(distance_matrix.shape[0], subsample_size, replace=False)
            distance_matrix = distance_matrix[indices][:, indices]
            labels = [0]*len(train) + [1]*len(test) + [2]*len(val)
            labels = np.array(labels)[indices]
            print(distance_matrix.shape)
        else:
            labels = [0]*len(train) + [1]*len(test) + [2]*len(val)
            labels = np.array(labels)
        reducer = umap.UMAP(metric='precomputed', random_state=42)
        embedding = reducer.fit_transform(distance_matrix)
        
        plt.figure(figsize=(10, 8), dpi=300)
        for label in np.unique(labels):
            selected = embedding[labels == label]
            plt.scatter(selected[:, 0], selected[:, 1], s=0.2, alpha=0.5, label=['Train', 'Test', 'Val'][label], marker='+')
        plt.title(f'{task_name} UMAP Embedding of Train-Test Distances')
        # indicate which color is which split and increase legend marker size
        plt.legend(markerscale=10)
        plt.savefig(f'data/{task_name}/umap_embedding.png')
#%%
    exit(0)
        
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

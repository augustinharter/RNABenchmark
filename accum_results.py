import json
import os
import sys

import numpy as np

model = 'BEACON-B/rnalm'
result_file_name = "test_results.json"

task_to_metric = {
    'Secondary_structure_prediction': 'eval_f1',
    'ContactMap': 'top_l_precision',
    'DistanceMap': 'r^2',
    'StructuralScoreImputation': 'eval_r^2',
    'SpliceAI': 'eval_acceptor topk acc',
    'NoncodingRNAFamily': 'eval_accuracy',
    'Modification': 'eval_mean_auc',
    'MeanRibosomeLoading': 'eval_r^2',
    'Degradation': 'MeanColumnWiseRootMeanSquaredError',
    'ProgrammableRNASwitches': 'eval_r^2_mean',
    'CRISPROnTarget': 'eval_spearman',
    'CRISPROffTarget': 'eval_spearman',
    'Isoform': 'eval_r^2',
}

no_seed_path_tasks = ['ContactMap', "CRISPROffTarget", 'DistanceMap', 'Isoform', 'Secondary_structure_prediction', 'SpliceAI', 'StructuralScoreImputation']
seed = 666
seed_path_tasks = ['ProgrammableRNASwitches', 'CRISPROnTarget', 'NoncodingRNAFamily', 'Modification', 'MeanRibosomeLoading']

paths = sys.argv[1:]    
all_results = {}
path = os.path.join(path, 'rna-all')
for task in os.listdir(path):
    if task == 'Degradation':
        # special case with MeanColumnWiseRootMeanSquaredError metric
        file_path = os.path.join(path, task, model, f'{seed}/results/rnalm_/submission_{seed}.csv')
        results = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=[2,3,4,5])
        columnwise_rmse = np.sqrt(np.mean(results**2, axis=0))
        mean_columnwise_rmse = np.mean(columnwise_rmse)
        all_results[task] = mean_columnwise_rmse
        continue

    # other options
    if task in no_seed_path_tasks:
        resultspath = 'results/rnalm_'
    elif task == "Modification":
        resultspath = f'{seed}/results/rnalm__seed{seed}_lr3e-5'
    else:
        resultspath = f'{seed}/results/rnalm_'

    eval_file = os.path.join(path, task, model, resultspath, result_file_name)
    if os.path.exists(eval_file):
        with open(eval_file, 'r') as f:
            try:
                results : dict = json.load(f)
                if not results:
                    all_results[task] = 0
                else:
                    metric = results[task_to_metric[task]]
                    all_results[task] = metric
            except Exception as e:
                print(f"Error processing results for {task} at {eval_file}: {e}")
    else:
        print(f"Results for {task}: No results found at {eval_file}")

print("Summary of results:")
for task, metric in all_results.items():
    print(f"{task}: {metric}")

#plot as bar chart
import matplotlib.pyplot as plt
tasks = list(all_results.keys())
metrics = [all_results[task] for task in tasks]
plt.figure(figsize=(10, 6))
plt.bar(tasks, metrics)
plt.xlabel('Task')
plt.ylabel('Metric')
plt.title('BEACON-B/RNAlm Performance on RNA Benchmark Tasks')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('beacon_b_rnalm_results.png')

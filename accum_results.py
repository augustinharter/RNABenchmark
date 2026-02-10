import json
import os
import sys

import numpy as np

model = 'BEACON-B/rnalm'
result_file_name = "test_results.json"

task_to_metric = {
    'Secondary_structure_prediction': 'f1',
    'ContactMap': 'top_l_precision',
    'DistanceMap': 'r^2',
    'StructuralScoreImputation': 'eval_r^2',
    'SpliceAI': 'eval_acceptor topk acc',
    'Isoform': 'eval_r^2',
    'NoncodingRNAFamily': 'eval_accuracy',
    'Modification': 'eval_mean_auc',
    'MeanRibosomeLoading': 'eval_r^2',
    'Degradation': 'MeanColumnWiseRootMeanSquaredError',
    'ProgrammableRNASwitches': 'eval_r^2_mean',
    'CRISPROnTarget': 'eval_spearman',
    'CRISPROffTarget': 'eval_spearman',
}

task_to_dataset_size = {
    'Secondary_structure_prediction': 10814,
    'ContactMap': 188,
    'DistanceMap': 188,
    'StructuralScoreImputation': 14049,
    'SpliceAI': 144628,
    'Isoform': 145463,
    'NoncodingRNAFamily': 5679,
    'Modification': 304661,
    'MeanRibosomeLoading': 76319,
    'Degradation': 2155,
    'ProgrammableRNASwitches': 73227,
    'CRISPROnTarget': 1453,
    'CRISPROffTarget': 14223,
}

# values from paper 64.18(0.44) 60.81(1.70) 56.28(0.41) 38.78(0.18) 37.43(1.43) 70.59(0.91) 94.63(0.16) 94.74(0.20) 72.29(0.28) 0.320(0.001) 54.67(0.36) 26.01(1.81) 4.42(0.33)

task_to_paper_results = {
    'Secondary_structure_prediction': 0.6418,
    'ContactMap': 0.6081,
    'DistanceMap': 0.5628,
    'StructuralScoreImputation': 0.3878,
    'SpliceAI': 0.3743,
    'Isoform': 0.7059,
    'NoncodingRNAFamily': 0.9463,
    'Modification': 0.9474,
    'MeanRibosomeLoading': 0.7229,
    'Degradation': 0.320,
    'ProgrammableRNASwitches': 0.5467,
    'CRISPROnTarget': 0.2601,
    'CRISPROffTarget': 0.0442,
}

no_seed_path_tasks = ['ContactMap', "CRISPROffTarget", 'DistanceMap', 'Isoform', 'Secondary_structure_prediction', 'SpliceAI', 'StructuralScoreImputation']
seed = 666
seed_path_tasks = ['ProgrammableRNASwitches', 'CRISPROnTarget', 'NoncodingRNAFamily', 'Modification', 'MeanRibosomeLoading']

limits = sys.argv[1:]    
all_results = {}
for limit in limits:
    limit_results = {}
    path = os.path.join('experiments', str(limit), 'rna-all')
    for task in os.listdir(path):
        try:
            if task == 'Degradation':
                # special case with MeanColumnWiseRootMeanSquaredError metric
                file_path = os.path.join(path, task, model, f'{seed}/results/rnalm_/submission_{seed}.csv')
                results = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=[2,3,4,5])
                columnwise_rmse = np.sqrt(np.mean(results**2, axis=0))
                mean_columnwise_rmse = np.mean(columnwise_rmse)
                limit_results[task] = mean_columnwise_rmse
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
                    results : dict = json.load(f)
                    if not results:
                        limit_results[task] = 0
                    else:
                        metric = results[task_to_metric[task]]
                        limit_results[task] = metric
            else:
                print(f"Results for {task}: No results found at {eval_file}")
        except Exception as e:
            print(f"Error processing results for {task} at {eval_file}: {e}")
    all_results[limit] = limit_results

# print("Summary of results:")
# for limit, results in all_results.items():
#     print(f"Limit: {limit}")
#     for task, metric in results.items():
#         print(f"{task}: {metric}")
#     print("\n")

#plot as bar chart, grou all limits together for each task
import matplotlib.pyplot as plt

width = 1 / (len(limits) + 1)  # width of each bar
spacing = 1.2

plt.figure(figsize=(3*len(limits), 6))

tasks = list(task_to_dataset_size.keys())
for lidx, limit in enumerate(limits):
    values = [100*all_results[limit][task] if task in all_results[limit] else 0 for task in tasks]
    x = [spacing*i + lidx * width for i in range(len(tasks))]
    bars =plt.bar(x, values, width=width, align='center', label=limit)
    # dont show leading 0s in bar labels
    plt.bar_label(bars, fmt='%.0f', padding=3, fontsize=8)

paper_values = [100*task_to_paper_results[task] for task in tasks]
x = [spacing*i + len(limits) * width for i in range(len(tasks))]
bars = plt.bar(x, paper_values, width=width, align='center', label='Paper')
plt.bar_label(bars, fmt='%.0f', padding=3, fontsize=8)

plt.legend(loc='upper left', ncols=len(limits)+1)
plt.xlabel('Task')
# disable yticks
plt.yticks([])
plt.title('Results with different fractions of data')
ticks = [f"{task}\n(n={task_to_dataset_size[task]})" for task in tasks]
plt.xticks(spacing*np.arange(len(tasks)) + width*len(limits), ticks, rotation=45, ha='right')
# disable tick lines
plt.tick_params(axis='x', which='both', length=0)
plt.tight_layout()
plt.savefig('limited_data_results.png')

import numpy as np
import torch
import random
from transformers import Trainer
import json
import os

def do_active_learning(model, complete_train_dataset, test_dataset, make_trainer_from_model_and_dataset):
    initial_fraction, iteration_fraction, num_iterations = [float(os.getenv('START_FRACTION')), float(os.getenv('ITERATION_FRACTION')), int(os.getenv('ITERATIONS'))]
    ranking_function = mc_dropout_ranking_function
    results_file_name = f'logs/al/mcdropout_spliceAI_{initial_fraction}_{iteration_fraction}_{num_iterations}.json'
    model_copy = model.state_dict()  # Save the initial state of the model to reset it in each iteration
    current_indices = random.sample(range(len(complete_train_dataset)), int(len(complete_train_dataset) * initial_fraction))
    current_train_set = torch.utils.data.Subset(complete_train_dataset, current_indices)

    active_learning_results = []
    for iteration in range(num_iterations):
        print(f'Active Learning Iteration {iteration+1}/{num_iterations} with current training fraction: {len(current_train_set)/len(complete_train_dataset):.2f}')
        model.load_state_dict(model_copy)  # Reset the model to its initial state for the next iteration
        trainer = make_trainer_from_model_and_dataset(model, current_train_set)
        trainer.train()
        results = trainer.evaluate(test_dataset)
        active_learning_results.append((f'{len(current_train_set)/len(complete_train_dataset):.2f}', results))
        print(f'Iteration {iteration+1} results: {results}')
        json.dump(active_learning_results, open(results_file_name, 'w'), indent=4)  # Save results after each iteration
        
        if iteration < num_iterations - 1:  # No need to get new indices after the last iteration
            next_indices = get_next_data_point_indices_for_active_learning(trainer, complete_train_dataset, current_indices, ranking_function, iteration_fraction)
            current_indices.extend(next_indices)
            current_train_set = torch.utils.data.Subset(complete_train_dataset, current_indices)


    return trainer, current_train_set


def get_next_data_point_indices_for_active_learning(trainer:Trainer, complete_train_dataset, current_indices, next_samples_selection_function, fraction_to_add):
    num_to_select_next = int(len(complete_train_dataset) * fraction_to_add)
    remaining_indices = list(set(range(len(complete_train_dataset))) - set(current_indices))
    #remaining_loader = torch.utils.data.DataLoader(remaining_dataset, batch_size=128, shuffle=False)
    best_inidices = next_samples_selection_function(trainer, complete_train_dataset, current_indices, remaining_indices, num_to_select_next)
    return best_inidices

def mc_dropout_ranking_function(trainer:Trainer, dataset, current_indices, remaining_indices, num_to_select_next, num_forward_passes=10):
    next_dataset = torch.utils.data.Subset(dataset, remaining_indices)
    trainer.model.train() # Enable dropout during inference
    trainer.args.disable_tqdm = True  # Disable tqdm progress bar for cleaner output during multiple forward passes
    loader = torch.utils.data.DataLoader(next_dataset, batch_size=128, shuffle=False)
    with torch.no_grad():
        uncertainties = []
        for i, batch in enumerate(loader):
            inputs = {k: v.to(trainer.args.device) for k, v in batch.items() if k != 'labels'}
            print(f'Batch {i+1}/{len(loader)}', end='\r')

            batch_logits = [[] for _ in range(inputs['input_ids'].shape[0])] # list of lists to store logits for each data point in the batch across forward passes
            for _ in range(num_forward_passes):
                outputs = trainer.model(**inputs)
                #print(outputs)
                for i in range(outputs.logits.shape[0]):
                    batch_logits[i].append(outputs.logits[i][inputs['attention_mask'][i].bool()])

            for i in range(inputs['input_ids'].shape[0]):
                sample_uncertainty = torch.var(torch.stack(batch_logits[i]), dim=0).mean().item()  # Variance across forward passes for this data point, avereged across tokens
                uncertainties.append(sample_uncertainty)
    
    print(f'Uncertainties calculated for {len(uncertainties)} data points.')
    print(f'Example uncertainties[:10]: {uncertainties[:10]}')  # Print the first 10 uncertainties for inspection

    best_indices_within_new_dataset = np.argsort(uncertainties)[-num_to_select_next:]
    best_inidices_within_complete_dataset = np.array(remaining_indices)[best_indices_within_new_dataset]
    return best_inidices_within_complete_dataset.tolist()

def coreset_ranking_function(trainer:Trainer, complete_train_dataset, current_indices, remaining_indices, num_to_select_next):
    current_dataset = torch.utils.data.Subset(complete_train_dataset, current_indices)
    remaining_dataset = torch.utils.data.Subset(complete_train_dataset, remaining_indices)
    
    current_logits = get_logits(trainer, current_dataset)
    remaining_logits = get_logits(trainer, remaining_dataset)
    print(remaining_logits.shape)
    # iteratively get the most distant point from the already selected ones
    selected_indices_within_remaing_set = []
    base_distances = torch.cdist(remaining_logits, current_logits, p=1)  # L1 distance between remaining points and current points
    current_closest_distances, _ = torch.min(base_distances, dim=1)  # distance to the closest point in the current set
    for _ in range(num_to_select_next):
        next_index = torch.argmax(current_closest_distances).item()
        selected_indices_within_remaing_set.append(next_index)
        # Update the distances to include the newly selected point
        new_distances = torch.cdist(remaining_logits, remaining_logits[next_index].unsqueeze(0), p=1).squeeze()  # distance to the newly selected point
        current_closest_distances = torch.min(current_closest_distances, new_distances, dim=0).values  # update the distance to the closest point in the current set
    best_inidices_within_complete_dataset = np.array(remaining_indices)[selected_indices_within_remaing_set]
    return best_inidices_within_complete_dataset.tolist()

def get_logits(trainer:Trainer, dataset):
    trainer.model.eval() # Disable dropout during inference
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(trainer.args.device) for k, v in batch.items() if k != 'labels'}
            outputs = trainer.model(**inputs)
            for i in range(outputs.logits.shape[0]):
                all_logits.append(outputs.logits[inputs['attention_mask'][i].bool()].cpu())
    return torch.cat(all_logits, dim=0).mean(dim=-2)  # average across tokens to get a single representation per data point


import numpy as np
import torch
import random
from transformers import Trainer
import json

def do_active_learning(model, complete_train_dataset, test_dataset, make_trainer_from_model_and_dataset, ranking_function, initial_fraction, iteration_fraction, num_iterations):
    model_copy = model.state_dict()  # Save the initial state of the model to reset it in each iteration
    current_indices = random.sample(range(len(complete_train_dataset)), int(len(complete_train_dataset) * initial_fraction))
    current_train_set = torch.utils.data.Subset(complete_train_dataset, current_indices)

    active_learning_results = []
    for iteration in range(num_iterations):
        print(f'Active Learning Iteration {iteration+1}/{num_iterations} with current training fraction: {len(current_train_set)/len(complete_train_dataset):.2f}')
        model.load_state_dict(model_copy)  # Reset the model to its initial state for the next iteration
        trainer = make_trainer_from_model_and_dataset(model, current_train_set)
        trainer.train()
        next_indices = get_next_data_point_indices_for_active_learning(trainer, complete_train_dataset, current_indices, ranking_function, iteration_fraction)
        current_indices.extend(next_indices)
        current_train_set = torch.utils.data.Subset(complete_train_dataset, current_indices)
        results = trainer.evaluate(test_dataset)
        active_learning_results.append((f'{len(current_train_set)/len(complete_train_dataset):.2f}', results))
        print(f'Iteration {iteration+1} results: {results}')
        json.dump(active_learning_results, open('active_learning_results.json', 'w'), indent=4)  # Save results after each iteration


    return trainer, current_train_set


def get_next_data_point_indices_for_active_learning(trainer:Trainer, complete_train_dataset, current_indices, active_learning_ranking_function, fraction_to_add):
    remaining_indices = list(set(range(len(complete_train_dataset))) - set(current_indices))
    remaining_dataset = torch.utils.data.Subset(complete_train_dataset, remaining_indices)
    #remaining_loader = torch.utils.data.DataLoader(remaining_dataset, batch_size=128, shuffle=False)

    scored_indices = active_learning_ranking_function(trainer, remaining_dataset)

    num_next_percentage = int(len(complete_train_dataset) * fraction_to_add)
    best_indices_indices = np.argsort(scored_indices)[-num_next_percentage:]
    best_inidices = np.array(remaining_indices)[best_indices_indices]
    return best_inidices

def mc_dropout_ranking_function(trainer:Trainer, dataset, num_forward_passes=2):
    trainer.model.train() # Enable dropout during inference
    trainer.args.disable_tqdm = True  # Disable tqdm progress bar for cleaner output during multiple forward passes
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    with torch.no_grad():
        uncertainties = []
        for i, batch in enumerate(loader):
            inputs = {k: v.to(trainer.args.device) for k, v in batch.items() if k != 'labels'}
            print(f'Batch {i+1}/{len(loader)}', end='\r')

            batch_logits = {i: [] for i in range(inputs['input_ids'].shape[0])}  # Store logits for each data point in the batch
            for _ in range(num_forward_passes):
                outputs = trainer.model(**inputs)
                #print(outputs)
                for i in range(outputs.logits.shape[0]):
                    batch_logits[i].append(outputs.logits[i][inputs['attention_mask'][i].bool()])

            for i in range(inputs['input_ids'].shape[0]):
                sample_ucnertainty = torch.var(torch.stack(batch_logits[i]), dim=0).mean().item()  # Variance across forward passes for this data point, avereged across tokens
                uncertainties.append(sample_ucnertainty)
    
    print(f'Uncertainties calculated for {len(uncertainties)} data points.')
    print(f'Example uncertainties[:10]: {uncertainties[:10]}')  # Print the first 10 uncertainties for inspection
    return uncertainties

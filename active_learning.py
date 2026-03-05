
import numpy as np
import torch
import random

def do_active_learning(model, complete_train_dataset, train_function, ranking_function, initial_fraction, iteration_fraction, num_iterations):
    model_copy = model.copy()
    current_indices = random.sample(range(len(complete_train_dataset)), int(len(complete_train_dataset) * initial_fraction))
    current_train_set = torch.utils.data.Subset(complete_train_dataset, current_indices)
    for iteration in range(num_iterations):
        model = model_copy.copy()  # Reset the model to its initial state for the next iteration
        model = train_function(model, current_train_set)
        next_indices = get_next_data_point_indices_for_active_learning(model, complete_train_dataset, current_indices, ranking_function, iteration_fraction)
        current_indices.extend(next_indices)
        current_train_set = torch.utils.data.Subset(complete_train_dataset, current_indices)

    return model, current_train_set


def get_next_data_point_indices_for_active_learning(model, complete_train_dataset, current_indices, ranking_function, fraction_to_add):
    remaining_indices = list(set(range(len(complete_train_dataset))) - set(current_indices))
    remaining_dataset = torch.utils.data.Subset(complete_train_dataset, remaining_indices)
    remaining_loader = torch.utils.data.DataLoader(remaining_dataset, batch_size=128, shuffle=False)

    ranked_indices = []
    for batch in remaining_loader:
        x = batch[0]  # Assuming the data is in the first element of the batch
        x = x.to(model.device)  # Move data to the same device as the model
        ranks = ranking_function(model, x)
        ranked_indices.extend(ranks)

    num_next_percentage = int(len(complete_train_dataset) * fraction_to_add)
    best_indices_indices = np.argsort(ranked_indices)[-num_next_percentage:]
    best_inidices = np.array(remaining_indices)[best_indices_indices]
    return best_inidices

def mc_dropout_ranking_function(model, x_batch, num_forward_passes=10):
    if not model.training:
        model.train()  # Ensure the model is in training mode to enable dropout
    with torch.no_grad():
        predictions = []
        for _ in range(num_forward_passes):
            preds = model(x_batch)
            predictions.append(preds)
    
    torch.stack(predictions)  # Shape: (num_forward_passes, batch_size, ...)
    uncertainty = torch.var(torch.stack(predictions), dim=0).flatten(start_dim=1).mean(dim=1)  # Variance across forward passes
    return uncertainty

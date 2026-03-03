
def get_next_data_set(complete_dataset, current_indices, ranking_function):
    remaining_indices = list(set(range(len(complete_dataset))) - set(current_indices))
    ranked_indices = ranking_function(complete_dataset, remaining_indices)
    next_index = ranked_indices[0]
    return next_index
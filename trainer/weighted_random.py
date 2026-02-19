import random
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader, TensorDataset


def weight_index(task_list):
    """
    For the given list of DataLoaders, perform weighted random sampling based on the number of samples in their datasets,
    and return the selected DataLoader.
    """
    # Ensure task_list contains datasets with length attributes
    weights = [len(task_loader) for task_loader in task_list]

    total_samples = sum(weights)
    probs = [w / total_samples for w in weights]

    chosen_index = random.choices(range(len(task_list)), weights=probs, k=1)[0]

    return chosen_index, task_list[chosen_index]


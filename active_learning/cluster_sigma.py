import torch
from active_learning.dynamic_threshold import dynamic_threshold
from active_learning.expected_model_change import expected_model_change
from active_learning.active_learning_cluster import (
    euclidean_distance, decision_graph, split_learning_pool
)


def cluster_sigma(model: callable, task_index: int,
                  train_samples_x: torch.Tensor, alpha: float = 0.1, lower_bound: list = None, upper_bound: list = None,
                  num: int = 10, split_ratio_1: float = 0.2, split_ratio_2: float = 0.4,  method: str = 'sigma_method'):
    size_train_x, dim_train_x = train_samples_x.shape

    # 生成样本池
    lower_bound_tensor = torch.tensor(lower_bound).reshape(1, -1)
    upper_bound_tensor = torch.tensor(upper_bound).reshape(1, -1)
    num_samples = 10000
    lower_bound_tensor = lower_bound_tensor.repeat(num_samples, 1)
    upper_bound_tensor = upper_bound_tensor.repeat(num_samples, 1)
    pool_samples = torch.rand(size=(num_samples, dim_train_x))
    pool_samples = pool_samples * (upper_bound_tensor - lower_bound_tensor) + lower_bound_tensor

    clusters, centers = split_learning_pool(model=model, learning_pool=pool_samples, num_cluster=num,
                                            split_ratio_1=split_ratio_1, split_ratio_2=split_ratio_2,
                                            task_id=task_index)

    for i in range(len(clusters)):
        cluster = clusters[i]
        threshold = dynamic_threshold(alpha, train_samples_x, lower_bound, upper_bound)
        if method == 'sigma_method':
            sigma = model(task_index, cluster).predictive.covariance.sqrt().detach().reshape(-1)
        elif method == 'expected_method':
            sigma = expected_model_change(model, task_index, cluster).detach().reshape(-1)

        for j, sample in enumerate(cluster):
            is_valid = True
            for train_samples in train_samples_x:
                abs_error = torch.abs(sample - train_samples)
                if (abs_error < threshold).any():
                    is_valid = False
                    break

            if not is_valid:
                sigma[j] = 0

        max_sigma, index_sigma = torch.max(sigma, dim=0)
        new_sample = cluster[index_sigma]
        if not (train_samples_x == new_sample).all(dim=1).any():
            train_samples_x = torch.cat([train_samples_x, new_sample.reshape(1, -1)], dim=0)

    return train_samples_x

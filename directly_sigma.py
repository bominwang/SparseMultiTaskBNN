import math
import torch
from active_learning.euclidean_distance import euclidean_distance


def directly_sigma(model: callable, task_index: int,
                   train_samples_x: torch.Tensor, lower_bound: list, upper_bound: list, threshold_n: int = 10):
    size_train_x, dim_train_x = train_samples_x.shape
    # 求解d_max
    lower_bound = torch.tensor(lower_bound).reshape(1, -1)
    upper_bound = torch.tensor(upper_bound).reshape(1, -1)
    diff = upper_bound - lower_bound
    d_max = torch.max(diff)

    # 生成池样本
    num_samples = 10000
    lower_bound = lower_bound.repeat(num_samples, 1)
    upper_bound = upper_bound.repeat(num_samples, 1)
    pool_samples = torch.rand(size=(num_samples, dim_train_x))
    true_pool_samples = pool_samples * (upper_bound - lower_bound) + lower_bound
    # 计算预测不确定性
    predictions = model(task_index, true_pool_samples).predictive
    pool_sigma = predictions.covariance.sqrt().detach().squeeze()
    # 定位最大后验概率
    k = math.floor(0.1 * num_samples)
    top_sigma, index = torch.topk(pool_sigma, k, dim=0, largest=True, sorted=True)
    top_samples = true_pool_samples[index]

    distance = euclidean_distance(train_samples_x, top_samples)
    min_distance = torch.min(distance, dim=1).values
    # 求解d_c
    d_c = torch.max(distance)
    # 确定阈值
    d_t = 0.01 * (d_max + d_c) / 2
    # 筛选样本
    count = 0

    for i in range(len(top_samples)):
        if min_distance[i] > d_t:
            count += 1
            train_samples_x = torch.cat([train_samples_x, top_samples[i].reshape(1, -1)], dim=0)
            # 求解欧式距离
            distance = euclidean_distance(train_samples_x, top_samples)
            min_distance = torch.min(distance, dim=1).values
            # 求解d_c
            d_c = torch.max(distance)
            # 确定阈值
            d_t = 0.1 * (d_max + d_c) / 2
        if count == threshold_n:
            break

    return train_samples_x

import torch
from active_learning.dynamic_threshold import dynamic_threshold
from active_learning.expected_model_change import expected_model_change

def euclidean_distance(point_set_1: torch.Tensor, point_set_2: torch.Tensor,
                       p: int = 2, normalized: bool = False):
    """
    计算point_set_1中的点到point_set_2中每一个点的最小欧式距离
    用于构建聚类中心，point_set_1为候选聚类中，point_set_2为已被选取的聚类中心
    """
    euclidean = torch.cdist(x1=point_set_1, x2=point_set_2, p=p)
    euclidean = torch.min(euclidean, dim=1)[0]
    if normalized:
        euclidean = euclidean / torch.max(euclidean)
    return euclidean


def decision_graph(x: torch.Tensor, x_std: torch.Tensor, centers: torch.Tensor, p: int = 2, normalized: bool = False):
    """
    决策图算法，用于选择新的中心点。

    参数:
    - x: 样本点的集合，形状为 (N, D)，其中 N 是样本数，D 是维度数。
    - x_std: 每个样本的标准差或不确定性，形状为 (N,)。
    - centers: 当前已选择的中心点集合，形状为 (M, D)，其中 M 是已选择的中心点数。
    - p: 距离度量的幂次，用于欧几里得距离或更高次范数的计算（默认为 2，即欧几里得距离）。
    - normalized: 是否对距离进行归一化处理（默认为 False）。

    返回:
    - new_center: 选定的新中心点，形状为 (D,)。
    - index: 选定新中心点在 x 中的索引。
    """
    num_x, dim_x = x.shape
    distances = euclidean_distance(x, centers, p=p, normalized=normalized)
    distances = distances.squeeze()
    x_std = x_std.squeeze()
    x_std = x_std / torch.max(x_std)
    combined_scores = distances + x_std
    index = torch.argmax(combined_scores)
    new_center = x[index]
    return new_center.reshape(1, -1), index


def split_learning_pool(model: callable, learning_pool: torch.Tensor, num_cluster: int,
                        split_ratio_1: float, split_ratio_2: float, task_id: int = None):
    """
    将learning_pool按照split_ratio_1和split_ratio_2的比例分割成三个子集
    其中属于前百分之split_ratio_1的样本为聚类中心候选样本，前百分之split_ratio_2的样本为聚类候选样本
    """
    assert split_ratio_2 >= split_ratio_1, "split_ratio_2 must be greater than or equal to split_ratio_1"
    assert split_ratio_1 >= 0 and split_ratio_2 <= 1, "split_ratio_1 and split_ratio_2 must be between 0 and 1"

    num_samples, num_dim = learning_pool.shape
    model.eval()
    if task_id is not None:
        predictions = model(task_id, learning_pool).predictive
    else:
        predictions = model(learning_pool).predictive

    std_pred = torch.sqrt(predictions.covariance.squeeze()).detach()
    sorted_std, sorted_index = torch.sort(std_pred, descending=True)
    learning_pool = learning_pool[sorted_index]
    # 聚类中心候选样本
    centre_pool = learning_pool[:int(num_samples * split_ratio_1)]
    centre_pool_std = sorted_std[:int(num_samples * split_ratio_1)]
    # 聚类候选样本
    candidate_pool = learning_pool[:int(num_samples * split_ratio_2)]
    # 选取聚类中心
    center_1 = centre_pool[0].reshape(1, num_dim)
    centers = center_1.clone()
    # 从候选样本池中将该点移除
    centre_pool = centre_pool[1:]
    centre_pool_std = centre_pool_std[1:]
    for i in range(num_cluster - 1):
        # 生成新的聚类中心
        new_center, new_index = decision_graph(x=centre_pool, x_std=centre_pool_std,
                                               centers=centers, p=2, normalized=True)
        # 将新聚类中心从centre_pool中删除
        centre_pool = torch.cat((centre_pool[:new_index], centre_pool[new_index + 1:]), dim=0)
        centre_pool_std = torch.cat((centre_pool_std[:new_index], centre_pool_std[new_index + 1:]), dim=0)
        # 将新聚类中心添加到centers中
        centers = torch.cat((centers, new_center.reshape(1, num_dim)), dim=0)

    # 执行聚类
    distance = torch.cdist(candidate_pool, centers, p=2)
    # 对于每一行，找到最小索引，即样本聚类中心索引
    _, closest_center_idx = torch.min(distance, dim=1)
    # 根据 closest_center_idx 分配样本
    clusters = [[] for _ in range(num_cluster)]
    # 分配样本
    for i in range(len(candidate_pool)):
        cluster_idx = closest_center_idx[i].item()
        clusters[cluster_idx].append(candidate_pool[i])

    # 将每个簇置为tensor
    clusters = [torch.stack(cluster) if len(cluster) > 0 else torch.empty([0, num_dim]) for cluster in clusters]
    return clusters, centers


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

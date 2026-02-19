import torch


def euclidean_distance(train_points, unpicked_points):
    """
    计算欧几里得距离
    :param train_points: 训练样本
    :param unpicked_points: 未选择样本
    """
    size_train, dim_train = train_points.shape
    size_unpicked, dim_unpicked = unpicked_points.shape

    unpicked_points = unpicked_points.unsqueeze(1).repeat(1, size_train, 1)
    train_points = train_points.unsqueeze(0).repeat(size_unpicked, 1, 1)

    per_distance = torch.pow(unpicked_points - train_points, 2)
    per_distance = torch.sum(per_distance, dim=2)
    return torch.sqrt(per_distance)

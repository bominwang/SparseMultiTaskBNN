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


def unique_function(train_points, unpicked_points):
    unique_unpicked_points = []
    for u_point in unpicked_points:
        if not any(torch.equal(u_point, t_point) for t_point in train_points):
            unique_unpicked_points.append(u_point)

    return torch.stack(unique_unpicked_points)


def inverse_distance_weights(train_points, unpicked_points):
    unpicked_points = unique_function(train_points, unpicked_points)
    euclidean_distances = euclidean_distance(train_points, unpicked_points)
    euclidean_distances_2 = euclidean_distances ** 2
    weights = torch.exp(- euclidean_distances_2) / euclidean_distances_2
    weights = torch.sum(weights, dim=1)
    return weights




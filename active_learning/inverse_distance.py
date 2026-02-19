import torch
from active_learning.euclidean_distance import euclidean_distance


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




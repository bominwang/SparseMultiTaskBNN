import torch


def dynamic_threshold(alpha: float, selected_samples: torch.Tensor, lower_bound: list, upper_bound: list):
    # 计算当前任务维度
    size_selected_samples, dim_selected_samples = selected_samples.shape
    lower_bound = torch.tensor(lower_bound).reshape(-1, 1)
    upper_bound = torch.tensor(upper_bound).reshape(-1, 1)
    box_bound = torch.abs(upper_bound - lower_bound)
    thresholds = torch.empty(dim_selected_samples, 1)
    for i in range(dim_selected_samples):
        selected_samples_idx = selected_samples[:, i].reshape(-1, 1)
        # 计算逐点距离
        pairwise_distance = torch.abs(selected_samples_idx - selected_samples_idx.unsqueeze(1))
        # 计算逐点距离的均值和标准差
        mean_distance = torch.mean(pairwise_distance)
        # std_distance = torch.std(pairwise_distance)
        # nom = mean_distance + 0.5 * std_distance + box_bound[i] + 1e-8
        # thresholds[i] = alpha * ((mean_distance + 0.5 * std_distance) / nom) * box_bound[i]
        nom = mean_distance + box_bound[i] + 1e-8
        thresholds[i] = alpha * (mean_distance / nom) * box_bound[i]

    return thresholds.reshape(1, -1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    lower_bound = [0, 0]
    upper_bound = [1, 1]
    selected_samples_sparse = torch.tensor([
        [0.1, 0.2], [0.9, 0.8], [0.5, 0.6], [0.2, 0.9], [0.7, 0.4],
        [0.3, 0.1], [0.8, 0.7], [0.4, 0.3], [0.6, 0.5], [0.2, 0.6],
        [0.4, 0.8], [0.7, 0.1], [0.9, 0.2], [0.5, 0.7], [0.3, 0.4]
    ])

    result_1 = dynamic_threshold(0.1, selected_samples_sparse, [0, 0], [1, 1])
    thresholds_x_1 = result_1[:, 0]
    thresholds_y_1 = result_1[:, 1]

    result_2 = dynamic_threshold(0.2, selected_samples_sparse, [0, 0], [1, 1])
    thresholds_x_2 = result_2[:, 0]
    thresholds_y_2 = result_2[:, 1]

    result_3 = dynamic_threshold(0.3, selected_samples_sparse, [0, 0], [1, 1])
    thresholds_x_3 = result_3[:, 0]
    thresholds_y_3 = result_3[:, 1]

    result_4 = dynamic_threshold(0.4, selected_samples_sparse, [0, 0], [1, 1])
    thresholds_x_4 = result_4[:, 0]
    thresholds_y_4 = result_4[:, 1]

    result_5 = dynamic_threshold(0.2, selected_samples_sparse, [0, 0], [1, 1])
    thresholds_x_5 = result_5[:, 0]
    thresholds_y_5 = result_5[:, 1]

    plt.figure(figsize=(8, 8))
    plt.scatter(selected_samples_sparse[:, 0], selected_samples_sparse[:, 1], color="blue", label="Sample Points")

    thresholds = {
        0.1: (thresholds_x_1, thresholds_y_1),
        0.2: (thresholds_x_2, thresholds_y_2),
        0.3: (thresholds_x_3, thresholds_y_3),
        0.4: (thresholds_x_4, thresholds_y_4),
        0.5: (thresholds_x_5, thresholds_y_5)
    }

    # Define colors for each alpha
    colors = {
        0.1: '#FF6347',  # Red
        0.2: '#4682B4',  # SteelBlue
        0.3: '#32CD32',  # LimeGreen
        0.4: '#FFD700',  # Gold
        0.5: '#8A2BE2'  # BlueViolet
    }

    plt.figure(figsize=(8, 8))
    plt.scatter(selected_samples_sparse[:, 0], selected_samples_sparse[:, 1], color="blue", label="Sample Points")

    # Loop over each alpha and plot the respective rectangles with correct color and label
    for alpha, (threshold_x, threshold_y) in thresholds.items():
        for (x, y) in selected_samples_sparse.numpy():
            rectangle = plt.Rectangle(
                (x - threshold_x / 2, y - threshold_y / 2),  # Bottom-left corner
                threshold_x,  # Width
                threshold_y,  # Height
                edgecolor=colors[alpha],  # Color corresponding to alpha
                facecolor="none",  # No fill
                linewidth=2  # Border thickness
            )
            plt.gca().add_patch(rectangle)

    # Adjust plot settings
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("X", fontsize=16)
    plt.ylabel("Y", fontsize=16)

    # Add a dynamic legend based on alpha values
    plt.legend(
        ["Current Training Dataset"] + [f"$\\alpha_{{base}}={alpha}$" for alpha in sorted(thresholds.keys())],
        fontsize=16
    )
    plt.axis("equal")
    plt.show()
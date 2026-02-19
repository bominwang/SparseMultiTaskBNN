import torch


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


def threshold_distances(Sample: torch.Tensor, ranges: list, alpha: float):
    """
    计算多维样本的阈值距离。

    参数:
    - Sample: 样本集合，形状为 (N, D)，其中 N 是样本数，D 是维度数。
    - ranges: 每个维度的上下界范围 [(l1, u1), (l2, u2), ...]。
    - alpha: 缩放因子 α_base，用于调整阈值大小。

    返回:
    - d_threshold: 每个维度的阈值距离 (D,)。
    """
    num_sample, dim_sample = Sample.shape
    absolute_distances = torch.tensor([abs(u - l) for u, l in ranges]).reshape(1, dim_sample)
    avg_distances = []

    for dim in range(dim_sample):
        # 当前维度样本
        dim_samples = Sample[:, dim].reshape(-1, 1)
        # 两两距离计算 (pairwise distances)
        pairwise_distances = torch.abs(dim_samples - dim_samples.T)
        avg_distance = pairwise_distances.sum() / (num_sample * (num_sample - 1))  # 平均距离
        avg_distances.append(avg_distance)

    avg_distances = torch.tensor(avg_distances).reshape(1, dim_sample)
    range_factors = avg_distances / (absolute_distances + avg_distances)
    d_threshold = alpha * avg_distances * range_factors * absolute_distances
    print(f'The distance threshold is: {d_threshold}')
    return d_threshold


def violate_index(pool: torch.Tensor, sample: torch.Tensor, threshold: torch.Tensor):
    abs_distances = torch.abs(pool.unsqueeze(1) - sample.unsqueeze(0))
    violation = abs_distances < threshold
    violation_mask = violation.any(dim=2).any(dim=1)
    violation_indices = torch.nonzero(violation_mask, as_tuple=False).flatten().tolist()
    return violation_indices


def maximum_sigma(Model: callable, Sample: torch.Tensor, Pool: torch.Tensor, d_threshold: torch.Tensor):
    num_pool, dim_pool = Pool.shape
    Model.eval()
    prediction = Model(Pool).predictive
    batch_std = torch.sqrt(prediction.covariance.squeeze()).detach().reshape(-1, 1)

    # 检测样本是否违反距离约束
    indices = violate_index(Pool, Sample, d_threshold)
    batch_std[indices] = 0
    max_std_index = torch.argmax(batch_std).item()
    max_std_sample = Pool[max_std_index, :].reshape(1, dim_pool)

    return max_std_sample, batch_std


def batch_maximum_sigma(Model: callable, Sample: torch.Tensor, Clusters: list, alpha: float, ranges: list):
    _, dim = Sample.shape
    d_threshold = threshold_distances(Sample, ranges, alpha)
    new_samples = []
    for idx in range(len(Clusters)):
        cluster = Clusters[idx]
        new_sample, new_std = maximum_sigma(Model, Sample, cluster, d_threshold)
        new_samples.append(new_sample)
    return torch.stack(new_samples, dim=0).reshape(-1, dim)


def expected_model_change(Model: callable, Sample: torch.Tensor, Pool: torch.Tensor, num_samples: int,
                          d_threshold: torch.Tensor, flag: bool = False, task_id: int = None):
    """
    计算期望模型改变：结合梯度的均值和方差来选择最有价值的样本
    """

    def calculate_gradient_norm(gradients):
        """计算梯度的L2范数"""
        norm_2 = 0.0
        for param_name, grad in gradients.items():
            norm_2 += grad.norm(2).item() ** 2
        return norm_2 ** 0.5

    # 得到预测分布的均值和标准差
    Model.eval()
    size, dim = Pool.shape
    if task_id is not None:
        prediction = Model(task_id, Pool).predictive
    else:
        prediction = Model(Pool).predictive

    batch_mean = prediction.mean.detach()
    batch_std = torch.sqrt(prediction.covariance.squeeze()).detach().reshape(-1, 1)
    pseudo_labels = torch.randn(size=[size, num_samples])
    pseudo_labels = batch_mean.repeat(1, num_samples) + batch_std.repeat(1, num_samples) * pseudo_labels

    Model.train()
    expected_grads = torch.zeros(size=[size, 1])
    grad_variance = torch.zeros(size=[size, 1])
    for i in range(size):
        sample = Pool[i, :].reshape(1, -1)
        if task_id is not None:
            pseudo_outputs = Model(task_id, sample)
        else:
            pseudo_outputs = Model(sample)
        grad_norms = []
        for j in range(num_samples):
            grads = {}
            loss_ = pseudo_outputs.train_loss_fn(pseudo_labels[i, j].reshape(-1, 1))
            loss_.backward(retain_graph=True)

            for name, para in Model.named_parameters():
                if para.grad is not None:
                    grads[name] = para.grad.clone()
                else:
                    grads[name] = torch.zeros(size=[1, 1])

            Model.zero_grad()
            grad_norm = calculate_gradient_norm(grads)
            grad_norms.append(grad_norm)

        mean_grads = torch.mean(torch.tensor(grad_norms))
        std_grads = torch.std(torch.tensor(grad_norms))
        expected_grads[i] = mean_grads
        grad_variance[i] = std_grads

    violation_indexes = violate_index(Pool, Sample, d_threshold)
    expected_grads[violation_indexes] = 0
    grad_variance[violation_indexes] = 0

    if flag:
        total_grads = expected_grads + grad_variance
    else:
        total_grads = expected_grads

    index = torch.argmax(total_grads)
    new_ = Pool[index, :]

    return new_


def batch_expected_model_change(Model: callable, Sample: torch.Tensor, Clusters: list, num_samples: int,
                                alpha: float, ranges: list, flag: bool = False, task_id: int = None):
    d_threshold = threshold_distances(Sample=Sample, ranges=ranges, alpha=alpha)
    new_samples = []
    for idx in range(len(Clusters)):
        cluster = Clusters[idx]
        new_sample = expected_model_change(Model=Model, Sample=Sample, Pool=cluster, num_samples=num_samples,
                                           d_threshold=d_threshold, flag=flag, task_id=task_id)
        new_samples.append(new_sample)
    return torch.stack(new_samples)


def plot_cluster_1d(model: callable, samples: torch.Tensor, x_pool: torch.Tensor, y_pred_std: torch.Tensor,
                    centers: torch.Tensor, clusters: list, cmap='viridis',
                    task_id: int = None, new_samples: torch.Tensor or list = None):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    cluster_colors = [
        "#FF5733",  # 鲜橙红
        "#33FF57",  # 鲜嫩绿
        "#3357FF",  # 电蓝
        "#FF33A8",  # 鲜粉红
        "#FFBD33",  # 明黄色
        "#8C33FF",  # 深紫色
        "#33FFF2",  # 亮青色
        "#FF8C33",  # 橙黄
        "#57FF33",  # 鲜草绿
        "#A833FF"  # 紫罗兰
    ]
    ps = np.stack([x_pool.numpy().squeeze(), y_pred_std.numpy().squeeze()], axis=1)
    segments = np.stack((ps[:-1], ps[1:]), axis=1)

    line_segments = LineCollection(segments, cmap=cmap, norm=plt.Normalize(y_pred_std.min(), y_pred_std.max()),
                                   alpha=0.5, zorder=1)
    line_segments.set_array(y_pred_std.numpy())
    line_segments.set_linewidth(6)
    # 设置绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.add_collection(line_segments)

    # 填充曲线下方的背景颜色
    for i, cluster in enumerate(clusters):
        cluster_color = cluster_colors[i % len(cluster_colors)]
        # 绘制聚类中心（中空三角形）
        if task_id is not None:
            centers_std = torch.sqrt(
                model(task_id, centers[i, :].reshape(1, -1)).predictive.covariance.squeeze()).detach()
        else:
            centers_std = torch.sqrt(model(centers[i, :].reshape(1, -1)).predictive.covariance.squeeze()).detach()
        plt.scatter(centers[i], centers_std, marker='v',
                    facecolors='none', edgecolors=cluster_color, linewidths=3, s=150, alpha=1, zorder=2)

        # 获取聚类的 x 和 y 坐标
        cluster_x = cluster.squeeze().numpy()
        if task_id is not None:
            cluster_y = torch.sqrt(model(task_id, cluster).predictive.covariance.squeeze()).detach().numpy()
        else:
            cluster_y = torch.sqrt(model(task_id, cluster).predictive.covariance.squeeze()).detach().numpy()

        # 确保 x 和 y 按顺序排列
        sorted_indices = np.argsort(cluster_x)
        cluster_x = cluster_x[sorted_indices]
        cluster_y = cluster_y[sorted_indices]

        # 填充背景颜色
        ax.fill_between(cluster_x, 0, cluster_y, color=cluster_color, alpha=0.2,
                        label=f'Cluster {i + 1}', zorder=0)

    # 中空圆形点（samples）
    if task_id is not None:
        samples_std = torch.sqrt(model(task_id, samples).predictive.covariance.squeeze()).detach()
    else:
        samples_std = torch.sqrt(model(samples).predictive.covariance.squeeze()).detach()
    plt.scatter(samples, samples_std, marker='o', facecolors='none',
                edgecolors='black', linewidths=3, s=150, alpha=1, zorder=2)

    # 中空五边形点（new_samples）
    if new_samples is not None:
        if task_id is not None:
            new_samples_std = torch.sqrt(model(task_id, new_samples).predictive.covariance.squeeze()).detach()
        else:
            new_samples_std = torch.sqrt(model(new_samples).predictive.covariance.squeeze()).detach()
        plt.scatter(new_samples, new_samples_std, marker='^', facecolors='none',
                    edgecolors='purple', linewidths=3, s=200, alpha=1, zorder=3, label='New samples')

    # 添加中空三角形的图例
    plt.scatter([], [], marker='v', s=150, facecolors='none', edgecolors='black', linewidths=2, label='Cluster center')

    plt.ylim(y_pred_std.min() - (0.05 * y_pred_std.min()), 1.05 * y_pred_std.max())
    plt.xlabel('x', fontsize=16)
    plt.ylabel('$\sigma$', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.3)
    plt.show()


def active_learning(funcs: list,  # 函数列表
                    model: callable,  # 模型
                    available_tasks: list,  # 可用任务列表
                    ranges: dict,  # 函数范围字典
                    multi_task_dataset: callable = None,  # 多任务数据集
                    num_points: int = 10,  # 待选点数量
                    num_samples_pool: int = 5000,  # 池采样数量
                    alpha: float = 0.02,  # 防止冗余超参数
                    split_ratio_1=0.2,  # 聚类划分比例
                    split_ratio_2=0.4,  # 聚类划分比例
                    flag=True  # 方差是否计入
                    ):
    def get_pool_samples(init_samples, var_low, var_high, var_dim):
        # 输入缩放
        for i in range(var_dim):
            init_samples[:, i] = var_low[i] + (var_high[i] - var_low[i]) * init_samples[:, i]
        return init_samples

    model.eval()
    for task_id in available_tasks:
        task_func = funcs[task_id]
        low = ranges[task_id]['low']
        high = ranges[task_id]['high']
        dim = len(low)  # 任务i的输入维度
        pool = torch.rand(size=[num_samples_pool, dim])
        pool = get_pool_samples(pool, low, high, dim)
        bound = list(zip(low, high))
        inputs, outputs = multi_task_dataset.task_data[task_id]
        clusters, centers = split_learning_pool(model=model,
                                                learning_pool=pool,
                                                num_cluster=num_points,
                                                split_ratio_1=split_ratio_1,
                                                split_ratio_2=split_ratio_2,
                                                task_id=task_id)
        new_samples = batch_expected_model_change(Model=model,
                                                  Sample=inputs,
                                                  Clusters=clusters,
                                                  num_samples=20,
                                                  alpha=alpha,
                                                  ranges=bound,
                                                  flag=flag,
                                                  task_id=task_id)
        _, dim = inputs.shape
        new_samples = new_samples.float().reshape(-1, dim)

        new_labels = task_func(new_samples)
        inputs = torch.cat([inputs, new_samples], dim=0)
        outputs = torch.cat([outputs, new_labels], dim=0)
        multi_task_dataset.task_data[task_id] = (inputs, outputs)

    return multi_task_dataset

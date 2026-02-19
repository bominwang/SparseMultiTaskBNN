import torch
import torch.optim as optim
from tqdm import tqdm
from copy import deepcopy


def deepcopy_model_safe(model):
    # 检查并移除 _lock 属性
    if hasattr(model, "_lock"):
        original_lock = model._lock  # 暂存原始 _lock
        model._lock = None  # 临时移除

    # 执行深拷贝
    model_copy = deepcopy(model)

    # 恢复 _lock 属性
    if hasattr(model, "_lock"):
        model._lock = original_lock  # 恢复原始 _lock

    return model_copy


def pruning(model: callable, mask_matrix: dict, prune_ratio: float, device: torch.device):
    """
    :param model: 待剪枝模型
    :param mask_matrix: 模型当前任务i的掩码矩阵，仅支持对单个任务掩码进行修改
    :param prune_ratio:  剪枝率
    """
    updated_masks = {}
    for name, module in model.base_net.named_modules():
        if isinstance(module, torch.nn.Linear):
            # 提取权重和偏置
            weights = module.weight.detach().clone().to(device) if module.weight is not None else None
            bias = module.bias.detach().clone().to(device) if module.bias is not None else None
            # 生成新的初始化掩码矩阵
            new_weight_mask = torch.ones_like(weights, device=device)
            new_bias_mask = torch.ones_like(bias, device=device)
            # 检测当前掩码矩阵
            current_weight_mask = mask_matrix[f'{name}.weight'].to(device)
            current_bias_mask = mask_matrix[f'{name}.bias'].to(device)
            # 更新一下权重
            weights *= current_weight_mask
            bias *= current_bias_mask
            new_weight_mask *= current_weight_mask
            new_bias_mask *= current_bias_mask
            # 获取未被剪枝的权重和偏置的值
            readable_weights = weights[new_weight_mask != 0]
            readable_bias = bias[new_bias_mask != 0]
            # 获取权重和偏置的绝对值
            readable_weights_abs = torch.abs(readable_weights)
            readable_bias_abs = torch.abs(readable_bias)
            # 对可更新值进行排序
            sorted_weights_abs, _ = torch.sort(readable_weights_abs, descending=False)
            sorted_bias_abs, _ = torch.sort(readable_bias_abs, descending=False)
            # 获取被剪枝参数数量
            pruned_weight_num = int(readable_weights.numel() * prune_ratio)
            pruned_bias_num = int(readable_bias.numel() * prune_ratio)
            # 获取剪枝阈值
            prune_weight_threshold = sorted_weights_abs[pruned_weight_num]
            prune_bias_threshold = sorted_bias_abs[pruned_bias_num]
            # 更新权重和偏置掩码矩阵
            index_weights = torch.abs(weights) < prune_weight_threshold
            index_bias = torch.abs(bias) < prune_bias_threshold

            new_weight_mask[index_weights] = 0
            new_bias_mask[index_bias] = 0
            # 更新掩码矩阵
            updated_masks[f'{name}.weight'] = new_weight_mask
            updated_masks[f'{name}.bias'] = new_bias_mask

    return updated_masks


def get_sparse_network(model: callable, multi_task_loader: list, multi_test_loader: list,
                       prune_ratio: float, prune_epochs: int, prune_train_epochs: int,
                       prune_learn_rate: float, device: torch.device):

    # 任务数量
    num_tasks = model.num_task
    # 保存预训练网络
    model_warmup = deepcopy_model_safe(model)
    criterion = torch.nn.L1Loss()
    # step1: 剪枝
    # 外部掩码
    task_masks = {}
    for task_id in range(num_tasks):
        model = deepcopy_model_safe(model_warmup).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=prune_learn_rate, weight_decay=0)

        iter_MASK = []
        iter_MAE = []
        for _ in tqdm(range(prune_epochs), desc=f"Pruning Task {task_id}", unit="epoch", colour='blue'):

            current_task_mask = model.masks_matrix[task_id]
            updated_masks = None
            task_loader = multi_task_loader[task_id]
            test_loader = multi_test_loader[task_id]

            for epoch in range(prune_train_epochs):
                for inputs, targets in task_loader:
                    optimizer.zero_grad()
                    inputs, targets = inputs.to(device), targets.to(device)
                    predictions = model(task_id, inputs)
                    loss = predictions.train_loss_fn(targets)
                    loss.backward()
                    optimizer.step()

            # 剪枝
            updated_masks = pruning(model, current_task_mask, prune_ratio, device)
            iter_MASK.append(updated_masks)
            # 更新模型掩码
            model.masks_matrix[task_id] = updated_masks
            # 微调
            for epoch in range(prune_train_epochs):
                model.train()
                for inputs, targets in task_loader:
                    optimizer.zero_grad()
                    inputs, targets = inputs.to(device), targets.to(device)
                    predictions = model(task_id, inputs)
                    loss = predictions.train_loss_fn(targets)
                    loss.backward()
                    optimizer.step()

            # 检测精度
            model.eval()

            for inputs, targets in test_loader:
                with torch.no_grad():
                    inputs, targets = inputs.to(device), targets.to(device)
                    predictions = model(task_id, inputs).predictive
                    mean = predictions.mean
                    loss = criterion(mean, targets)
                    iter_MAE.append(loss.item())

        best_index = iter_MAE.index(min(iter_MAE))
        task_masks[task_id] = iter_MASK[best_index]

    model = deepcopy_model_safe(model)
    model.masks_matrix = task_masks
    # model.to(torch.device("cpu"))
    return model

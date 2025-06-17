import torch


def get_filtered_task_names(task_names, excluded_task):
    """
    获取不包含指定任务名称的任务列表。

    参数:
    - task_names (list): 包含所有任务名称的列表。
    - excluded_task (str): 需要排除的任务名称。

    返回:
    - list: 去除了指定任务的任务名称列表。

    功能说明:
    该函数用于在多任务学习中获取一个过滤后的任务列表，以便用于梯度手术过程中避免对同一任务进行自投影。
    """
    return [name for name in task_names if name != excluded_task]


def get_grad(grad_dict: dict):
    """
    合并所有任务的梯度，生成一个最终的梯度字典。

    参数:
    - grad_dict (dict): 每个任务的梯度字典，结构为 {task_name: {param_name: grad_tensor}}。

    返回:
    - dict: 包含所有任务累加梯度的字典，格式为 {param_name: combined_grad_tensor}。

    计算步骤:
    1. 初始化 `combined_grad` 字典，存储每个参数的累加梯度。
    2. 遍历 `grad_dict` 中的每个任务的梯度，将相同参数的梯度累加。
    3. 返回合并后的梯度字典 `combined_grad`。

    功能说明:
    该函数用于将多任务的梯度合并到一个字典中，以便用于更新模型参数。最终生成的梯度字典包含了所有任务的梯度合并结果。
    """
    combined_grad = {}
    for task_name, task_grad in grad_dict.items():
        for param_name, grad in task_grad.items():
            if param_name not in combined_grad:
                combined_grad[param_name] = torch.zeros_like(grad)
            combined_grad[param_name] += grad
    return combined_grad


def gradient_surgery(grad_dict: dict):
    """
    执行梯度手术 (PCGrad) 操作，以消除不同任务间的梯度冲突。

    参数:
    - grad_dict (dict): 每个任务的梯度字典，结构为 {task_name: {param_name: grad_tensor}}。

    返回:
    - dict: 执行梯度手术后的梯度字典，包含修正后的梯度。

    计算步骤:
    1. 获取所有任务名称列表 `task_names`。
    2. 对于每个任务 `task_name`，获取该任务的梯度 `main_grad`。
    3. 调用 `get_filtered_task_names` 获取除当前任务外的其他任务名称。
    4. 对于每个参数 `name`，检查其名称中是否包含 "base_net"（只处理 base_net 的梯度）。
        - 如果是，则将其展平为 `grad_flatten`，方便进行向量操作。
        - 遍历 `filtered_task_names` 中的其他任务梯度，对每个 `filtered_grad`：
            - 计算 `grad_flatten` 和 `filtered_grad` 的点积 `vector_dot`。
            - 如果 `vector_dot < 0`，说明存在梯度冲突，进行投影修正：
                \[
                \text{grad\_flatten} -= \frac{\text{vector\_dot}}{\|\text{filtered\_grad}\|^2 + 1e-8} \times \text{filtered\_grad}
                \]
        - 将修正后的 `grad_flatten` 还原为原始形状并更新回 `main_grad`。
    5. 返回修正后的梯度字典 `grad_dict`。

    功能说明:
    该函数用于在多任务学习中通过梯度投影减少任务之间的梯度冲突。每个任务的 `base_net` 梯度会与其他任务的梯度进行检查，如果存在方向冲突的梯度，将执行投影操作，以便最终更新时能更好地兼顾各任务。
    """
    task_names = list(grad_dict.keys())
    for task_name in task_names:
        main_grad = grad_dict[task_name]
        filtered_task_names = get_filtered_task_names(task_names, task_name)

        for name, grad in main_grad.items():
            if 'base_net' in name:
                grad_flatten = grad.view(-1)

                for filtered_task_name in filtered_task_names:
                    filtered_grad = grad_dict[filtered_task_name][name].view(-1)
                    vector_dot = torch.dot(grad_flatten, filtered_grad)

                    if vector_dot < 0:
                        grad_flatten -= vector_dot * filtered_grad / (filtered_grad.norm() ** 2 + 1e-8)

                grad_flatten = grad_flatten.view(grad.shape)
                main_grad[name] = grad_flatten

        grad_dict[task_name] = main_grad

    return grad_dict

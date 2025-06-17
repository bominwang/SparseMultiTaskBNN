import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import zip_longest
from trainer.weighted_random import weight_index
from trainer.gradient_surgery import gradient_surgery, get_grad


def PcGrad_Trainer(model: nn.Module, multi_task_loader: list, learning_rate: float, epochs: int, device: torch.device,
                   verbose: bool = True):
    """
    按照 PCGrad 思路进行“多任务小样本”训练，按任务交替处理每个 batch。

    每个 epoch 包含以下步骤：
          1) 同步迭代每个任务的数据加载器，按任务交替处理每个 batch。
          2) 对当前 batch 的所有任务进行 forward/backward，获取梯度。
          3) 执行梯度手术 (gradient_surgery)。
          4) 合并梯度 (get_grad)，再一次性更新模型 (optimizer.step)。
    """
    model.to(device)
    num_tasks = len(multi_task_loader)
    name_task = list(range(num_tasks))

    # 初始化每个任务的损失记录
    epoch_losses = {task_id: [] for task_id in name_task}

    # 定义优化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)

    # 定义损失函数（根据任务需求调整）
    loss_fns = {
        0: nn.MSELoss(),
        1: nn.CrossEntropyLoss(),
        2: nn.BCELoss()
        # 根据任务数量和类型继续添加
    }

    # 创建进度条
    progress_bar = tqdm(range(epochs), desc="PcGrad", unit="epoch", colour='green')

    for epoch in progress_bar:
        model.train()
        optimizer.zero_grad()

        # 初始化每个任务的数据加载器迭代器
        task_iterators = [iter(loader) for loader in multi_task_loader]

        # 初始化每个任务的损失和步数
        tasks_train_loss = {task_id: 0.0 for task_id in name_task}
        tasks_steps = {task_id: 0 for task_id in name_task}

        # 使用 zip_longest 迭代所有任务的 DataLoader，按任务交替处理
        for batches in zip_longest(*task_iterators, fillvalue=None):
            batch_data = {}
            for task_id, data in enumerate(batches):
                if data is not None:
                    inputs, targets = data
                    batch_data[task_id] = (inputs.to(device), targets.to(device))

            if not batch_data:
                break  # 所有任务的数据加载器均已耗尽，退出循环

            # ---- (1) 计算每个任务的梯度并累积
            accumulated_grads = {
                task_id: {
                    name: torch.zeros_like(param) for name, param in model.named_parameters()
                }
                for task_id in batch_data.keys()
            }

            for task_id, (inputs, targets) in batch_data.items():
                # 前向传播
                predictions = model(task_id, inputs)
                loss = predictions.train_loss_fn(targets)

                # 累加任务损失
                tasks_train_loss[task_id] += loss.item()
                tasks_steps[task_id] += 1

                # 反向传播
                loss.backward()

                # 保存当前任务的梯度
                for param_name, param in model.named_parameters():
                    if param.grad is not None:
                        accumulated_grads[task_id][param_name] += param.grad.clone()

                # 清除当前任务的梯度，以避免累积
                model.zero_grad()

            # ---- (2) PCGrad：对 accumulated_grads 进行梯度手术
            accumulated_grads = gradient_surgery(accumulated_grads)

            # ---- (3) 合并梯度并更新模型
            merged_grad = get_grad(accumulated_grads)
            for param_name, param in model.named_parameters():
                if param_name in merged_grad:
                    param.grad = merged_grad[param_name]

            # 进行梯度截断，防止梯度爆炸
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 更新模型参数
            optimizer.step()
            optimizer.zero_grad()

        # ---- (4) 记录各任务的平均损失
        for task_id in name_task:
            if tasks_steps[task_id] > 0:
                avg_loss = tasks_train_loss[task_id] / tasks_steps[task_id]
            else:
                avg_loss = 0.0
            epoch_losses[task_id].append(avg_loss)

        # ---- (5) 打印日志
        if verbose:
            loss_msg = " | ".join(
                f"Task {tid}: {epoch_losses[tid][-1]:.4f}" for tid in name_task
            )
            progress_bar.set_postfix_str(loss_msg)

    # 训练完成后，把模型移回 CPU
    model.to(torch.device("cpu"))

    # ---- (6) 可视化损失曲线
    if verbose:
        plt.figure(figsize=(10, 6))
        for task_id in name_task:
            plt.plot(range(1, epochs + 1), epoch_losses[task_id], label=f"Task {task_id} Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Loss")
        plt.title("PCGrad Multi-Task Training Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return model


def Random_Trainer(model: callable, multi_task_loader: list, learning_rate: float, epochs: int, device: torch.device,
                   verbose: bool = True):
    model.to(device)
    num_tasks = len(multi_task_loader)
    name_task = list(range(num_tasks))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)

    # Initialize loss storage
    tasks_train_loss = {tid: 0.0 for tid in name_task}
    tasks_steps = {tid: 0 for tid in name_task}
    epoch_losses = {tid: [] for tid in name_task}

    progress_bar = tqdm(range(epochs), desc="PcGrad", unit="batch", colour='green')
    for epoch in progress_bar:
        model.train()
        model.zero_grad()
        task_id, task_loader = weight_index(multi_task_loader)
        for inputs, targets in task_loader:
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(task_id, inputs)
            loss = predictions.train_loss_fn(targets)
            loss.backward()
            optimizer.step()
            # Accumulate loss for the task
            tasks_train_loss[task_id] += loss.item()
            tasks_steps[task_id] += 1

        # ---- (4) Calculate and store average loss per task
        for task_id in name_task:
            if tasks_steps[task_id] > 0:
                avg_loss = tasks_train_loss[task_id] / tasks_steps[task_id]
            else:
                avg_loss = 0.0
            epoch_losses[task_id].append(avg_loss)

        if verbose:
            loss_msg = " | ".join(
                f"Task {tid}: {epoch_losses[tid][-1]:.4f}" for tid in name_task
            )
            progress_bar.set_postfix_str(loss_msg)

    return  model.to(torch.device("cpu"))

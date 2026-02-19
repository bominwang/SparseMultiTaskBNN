import torch


def expected_model_change(Model: callable, task_id: int, Pool: torch.Tensor, num_samples: int = 10,):

    """
    得到样本池中样本的梯度信息
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
    prediction = Model(task_id, Pool).predictive

    batch_mean = prediction.mean.detach()
    batch_std = torch.sqrt(prediction.covariance.squeeze()).detach().reshape(-1, 1)

    pseudo_labels = torch.randn(size=[size, num_samples])
    pseudo_labels = batch_mean.repeat(1, num_samples) + batch_std.repeat(1, num_samples) * pseudo_labels

    Model.train()
    expected_grads = torch.zeros(size=[size, 1])
    grad_variance = torch.zeros(size=[size, 1])

    for i in range(size):
        sample = Pool[i, :].reshape(1, -1)
        pseudo_outputs = Model(task_id, sample)
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
        expected_grads[i] = mean_grads

    return expected_grads

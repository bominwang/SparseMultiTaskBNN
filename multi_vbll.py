import torch
import torch.nn as nn
from models.regression import Regression
import pprint
from models.decorator import info_train2model
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class BaseMlp(nn.Module):
    """
    过参数化的基础网络
    参数
    --------------------------------------------------------------------------------------------------------------------
    input_features: int 基础网络输入维度
    hidden_features: int 隐藏层维度
    num_layers: int 隐藏藏层数，需要大于1
    --------------------------------------------------------------------------------------------------------------------
    """

    def __init__(self, input_features: int, hidden_features: int, num_layers: int):
        super().__init__()
        assert input_features > 0, "input_features must be greater than 0"
        assert hidden_features > 0, "hidden_features must be greater than 0"
        assert num_layers > 1, "num_layers must be greater than 0"

        self.input_features = input_features
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        # 构建网络
        self.net = self.__built_net__()
        self.__initialize_weights__()

    def __built_net__(self):
        net = nn.Sequential()
        net.add_module('fc1', nn.Linear(self.input_features, self.hidden_features))
        net.add_module('elu1', nn.ELU())
        for i in range(self.num_layers - 1):
            net.add_module(f'fc{i + 2}', nn.Linear(self.hidden_features, self.hidden_features, bias=True))
            net.add_module(f'elu{i + 2}', nn.ELU())
        return net

    def __initialize_weights__(self):
        for layer in self.net.children():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.1)

    def forward(self, x):
        return self.net(x)


class TaskMlp(nn.Module):
    """
    任务分支网络
    参数
    --------------------------------------------------------------------------------------------------------------------
    input_features: int 任务分支网络的输入维度 （任务网络的实际输入为基础网络的输出维度加上该维度）
    output_features: int 任务分支网络的输入维度
    hidden_features: int 隐藏层维度
    num_layers: int 隐藏层层数
    prior_scale: float 协方差矩阵的精度先验
    reg_weight: float ELBO中的正则化项的权重（这一项是对预测不确定性程度影响最大的）
    wishart_scale: float 噪声协方差矩阵wishart先验的精度
    parameterization: str 协方差矩阵的参数化类型  {'dense', 'diagonal', 'lowrank', 'dense_precision'}
    dof: float wishart先验的自由度
    --------------------------------------------------------------------------------------------------------------------
    关于VBLL的细节可以查看：https://vbll.readthedocs.io/en/latest/index.html
    """

    def __init__(self, input_features: int, output_features: int, hidden_features: int, num_layers: int,
                 prior_scale: float, reg_weight: float, wishart_scale: float, parameterization: str, dof: float = 1.0):
        super().__init__()
        assert input_features >= 0, "input_features must be greater than or equal to 0"
        assert output_features > 0, "output_features must be greater than 0"
        assert hidden_features > 1, "hidden_features must be greater than of equal to 1"

        self.input_features = input_features
        self.output_features = output_features
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.prior_scale = prior_scale
        self.reg_weight = reg_weight
        self.wishart_scale = wishart_scale
        self.parameterization = parameterization
        self.dof = dof
        self.net = self.__built_net__()
        self.__initialize_weights__()

    def __built_net__(self):
        net = nn.Sequential()
        net.add_module('fc1', nn.Linear(self.input_features, self.hidden_features, bias=True))
        net.add_module('elu1', nn.ELU())
        for i in range(self.num_layers - 1):
            net.add_module(f'fc{i + 2}', nn.Linear(self.hidden_features, self.hidden_features, bias=True))
            net.add_module(f'elu{i + 2}', nn.ELU())
        net.append(Regression(in_features=self.hidden_features, out_features=self.output_features,
                              regularization_weight=self.reg_weight, parameterization=self.parameterization,
                              wishart_scale=self.wishart_scale, dof=self.dof, prior_scale=self.prior_scale))
        return net

    def __initialize_weights__(self):
        for layer in self.net.children():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return self.net(x)


@info_train2model
class MultiTaskVBLL(nn.Module):
    """
    多任务神经网络 (Multi-Task Neural Network) 的实现
    生成网络中包含一个共享的 BaseNet 和多个任务特定的 TaskNet， 其中TaskNet的最优一层为贝叶斯层
    参数：
    --------------------------------------------------------------------------------------------------------------------
        config_BaseNet: dict
        配置 BaseNet 的字典，包含以下键值对:
        - 'input_features': int, 输入特征的维度。
        - 'output_features': int, BaseNet 的输出特征维度 (即共享特征)。
        - 'num_hidden': int, BaseNet 中的层数。

        config_TaskNet: dict
        配置多个 TaskNet 的字典，包含以下键值对:
        - 'num_task': int, 任务的数量。
        - 'input_features': list, 每个任务特定的输入特征维度，与 BaseNet 的输出特征结合。
        - 'output_features': list, 每个任务的输出维度，可以是单一值或任务特定的列表。
        - 'hidden_features': int, 每个 TaskNet 隐藏层的特征维度。
        - 'num_hidden': list, 每个 TaskNet 的层数。
        - 'prior_scale': float 协方差矩阵的精度先验
        - 'reg_weight': float ELBO中的正则化项的权重（这一项是对预测不确定性程度影响最大的）
        - 'wishart_scale': float 噪声协方差矩阵wishart先验的精度
        - 'parameterization': str 协方差矩阵的参数化类型  {'dense', 'diagonal', 'lowrank', 'dense_precision'}
        - 'dof': float wishart先验的自由度
    """

    def __init__(self, config_BaseNet, config_TaskNet, debug=False):
        super().__init__()

        self.config_BaseNet = config_BaseNet
        self.config_TaskNet = config_TaskNet
        self.debug = debug

        # 构建共享网络
        self.base_net = self.__built_base_net__()

        # 构建分支网络
        self.num_task = config_TaskNet['num_task']
        self.config_TaskNet['input_features'] = [config_BaseNet['hidden_features'] + input_f
                                                 for input_f in config_TaskNet['input_features']]

        self.task_nets = self.__built_task_list__()

        # 预热后的网络参数
        self.warmup_weights = {}
        # 初始化掩码矩阵
        self.masks_matrix = self.get_mask_matrix()

        if self.debug:
            pprint.pprint(self)
            pprint.pprint(self.masks_basenet)
            print(self.mask_matrix)

    def __built_base_net__(self):
        base_net = BaseMlp(input_features=self.config_BaseNet['input_features'],
                           hidden_features=self.config_BaseNet['hidden_features'],
                           num_layers=self.config_BaseNet['num_hidden'])
        return base_net

    def __built_task_list__(self):
        task_list = nn.ModuleList()
        for task_id in range(self.num_task):
            task_list.append(TaskMlp(input_features=self.config_TaskNet['input_features'][task_id],
                                     output_features=self.config_TaskNet['output_features'][task_id],
                                     hidden_features=self.config_TaskNet['hidden_features'],
                                     num_layers=self.config_TaskNet['num_hidden'],
                                     prior_scale=self.config_TaskNet['prior_scale'],
                                     reg_weight=self.config_TaskNet['reg_weight'],
                                     wishart_scale=self.config_TaskNet['wishart_scale'],
                                     parameterization=self.config_TaskNet['parameterization'],
                                     dof=self.config_TaskNet['dof']))
        return task_list

    def get_mask_matrix(self):
        mask_matrix = {
            task_id: {name: torch.ones_like(param, requires_grad=False)
                      for name, param in self.base_net.named_parameters()}
            for task_id in range(self.num_task)
        }
        return mask_matrix

    def forward(self, task_id: int, x: torch.Tensor):
        x = x.float()
        common_features = x[:, :self.config_BaseNet['input_features']]
        specific_features = x[:, self.config_BaseNet['input_features']:]

        for name, module in self.base_net.named_children():
            if self.debug:
                print(f'{name}: {module}')
            if isinstance(module, nn.Linear):
                masked_weight = self.mask_matrix[task_id][f'{name}.weight'] * module.weight
                masked_bias = self.mask_matrix[task_id][f'{name}.bias'] * module.bias
                common_features = nn.functional.linear(input=common_features, weight=masked_weight, bias=masked_bias)
            else:
                common_features = module(common_features)
        features = torch.cat([common_features, specific_features], dim=1)
        # print(features.shape)
        if self.debug:
            print(f'features: {features}')

        y = self.task_nets[task_id](features)

        return y

    def plot_layer_mask_heatmaps(self):
        """
        为每一层绘制单独的掩码矩阵热力图，并标明层名称。

        返回：
        None
        """
        # 获取网络层的参数（包括掩码矩阵）
        layer_names = []
        all_mask_values = []

        # 获取每个任务的掩码矩阵并叠加
        for name, param in self.base_net.named_parameters():
            if name.endswith('.weight'):  # 只对权重参数进行操作
                layer_name = name.split('.')[0]  # 获取当前层的名字（比如 'fc1'）
                layer_names.append(layer_name)

                # 获取该层的掩码矩阵并将其叠加
                layer_mask = np.zeros_like(param.data.cpu().numpy())

                for task_id in range(self.num_task):
                    layer_mask += self.masks_matrix[task_id][name].cpu().numpy()

                all_mask_values.append(layer_mask)

        # 为每一层绘制独立的热力图
        for i, (layer_name, mask_values) in enumerate(zip(layer_names, all_mask_values)):
            plt.figure(figsize=(8, 6))
            sns.heatmap(mask_values, cmap='Blues', annot=False, cbar=True, xticklabels=False, yticklabels=False)
            plt.title(f'Mask Heatmap for Layer: {layer_name}')
            plt.tight_layout()
            plt.show()

    def analyze_mask_sparsity_and_overlap(self):
        """
        分析每个任务的掩码稀疏度和任务间掩码的重叠数。

        输出结果：
        - 每个任务的掩码稀疏度（即掩码中值为 1 的比例）。
        - 每两个任务之间的掩码重叠数。
        """
        # 存储每个任务的掩码稀疏度
        sparsity = {}

        # 存储任务间的掩码重叠数
        overlap_counts = {}

        # 计算每个任务的掩码稀疏度和任务间掩码的重叠数
        for task_id in range(self.num_task):
            sparsity[task_id] = {}
            total_mask_elements = 0
            active_mask_elements = 0

            # 遍历 BaseNet 中的每一层的掩码
            for name, param in self.base_net.named_parameters():
                if name.endswith('.weight'):  # 只对权重进行操作
                    mask = self.masks_matrix[task_id][name].cpu().numpy()
                    total_mask_elements += mask.size
                    active_mask_elements += np.sum(mask)

            sparsity[task_id] = active_mask_elements / total_mask_elements  # 计算稀疏度

        # 计算任务间掩码重叠数
        for task_id_1 in range(self.num_task):
            for task_id_2 in range(task_id_1 + 1, self.num_task):
                overlap_counts[(task_id_1, task_id_2)] = 0

                # 遍历 BaseNet 中的每一层的掩码
                for name, param in self.base_net.named_parameters():
                    if name.endswith('.weight'):  # 只对权重进行操作
                        mask_1 = self.masks_matrix[task_id_1][name].cpu().numpy()
                        mask_2 = self.masks_matrix[task_id_2][name].cpu().numpy()
                        # 计算两个掩码之间的重叠数
                        overlap_counts[(task_id_1, task_id_2)] += np.sum((mask_1 == 1) & (mask_2 == 1))

        # 格式化输出为表格形式
        output_data = []

        # 添加稀疏度数据
        for task_id in range(self.num_task):
            output_data.append(['Task', task_id, 'Sparsity', sparsity[task_id]])

        # 添加任务间重叠数数据
        for (task_id_1, task_id_2), overlap_count in overlap_counts.items():
            output_data.append(['Overlap', f'Task {task_id_1} & Task {task_id_2}', 'Count', overlap_count])

        # 使用 pprint 打印输出
        pprint.pprint(output_data)

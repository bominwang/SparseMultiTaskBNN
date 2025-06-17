def info_train2model(cls):
    """
    装饰器，用于向 PyTorch 模型类添加一个私有属性 `_info_task_data`，
    并提供该属性的访问（getter）和修改（setter）接口。

    主要用途：
    - 通过 `_info_task_data` 属性，可以为模型实例动态添加任务相关的信息或元数据。
    - 此属性可以在模型的训练、评估或其他过程中存储与任务相关的数据。
    - 提供一个结构化的方式将元数据嵌入到模型中，而不需要直接修改模型类。

    参数：
    cls (type): 要装饰的类，必须是 `torch.nn.Module` 的子类。

    返回：
    cls (type): 添加了 `_info_task_data` 属性的原始类。

    异常：
    - TypeError: 如果传入的类不是 `torch.nn.Module` 的子类，则抛出此异常。
    - AttributeError: 如果传入的类已定义 `_info_task_data` 或 `info_task_data` 属性，则抛出此异常。

    示例：
    ```python
    import torch.nn as nn

    @info_train2model
    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.layer = nn.Linear(10, 1)

    model = MyModel()

    # 使用新添加的 `_info_task_data` 属性
    model.info_task_data = {"task": "regression", "dataset": "example"}
    print(model.info_task_data)  # 输出：{'task': 'regression', 'dataset': 'example'}
    ```
    """
    import torch.nn as nn
    import threading

    # 检查传入的类是否是 `torch.nn.Module` 的子类
    if not issubclass(cls, nn.Module):
        raise TypeError("cls must be a subclass of nn.Module")

    # 检查类中是否已经存在 `_info_task_data` 或 `info_task_data`
    if hasattr(cls, "_info_task_data") or hasattr(cls, "info_task_data"):
        raise AttributeError("The class already defines '_info_task_data' or 'info_task_data'. Please choose a different attribute name.")

    # 保存原始的 __init__ 方法
    orig_init = cls.__init__

    # 定义新的 __init__ 方法
    def new_init(self, *args, **kwargs):
        """
        新的初始化方法：
        - 调用原始的 __init__ 方法以保持原始行为。
        - 为模型实例添加一个私有属性 `_info_task_data`，初始化为空字典。
        - 添加一个 `_lock` 属性，用于线程安全。
        """
        orig_init(self, *args, **kwargs)
        self._info_task_data = {}  # 初始化为一个空字典
        self._lock = threading.Lock()  # 用于线程安全

    # 定义 `_info_task_data` 的 getter 方法
    @property
    def info_task_data(self):
        """
        获取 `_info_task_data` 属性的值（线程安全）。
        """
        with self._lock:  # 使用线程锁保护读取操作
            return self._info_task_data

    # 定义 `_info_task_data` 的 setter 方法
    @info_task_data.setter
    def info_task_data(self, value):
        """
        设置 `_info_task_data` 属性的值（线程安全）。

        参数：
        value (dict): 新的值，必须是字典类型。
        """
        if not isinstance(value, dict):
            raise TypeError("info_task_data must be a dictionary.")
        with self._lock:  # 使用线程锁保护写入操作
            self._info_task_data = value

    # 替换类的原始 __init__ 方法为新的初始化方法
    cls.__init__ = new_init

    # 为类添加 `info_task_data` 属性（getter 和 setter）
    cls.info_task_data = info_task_data

    return cls

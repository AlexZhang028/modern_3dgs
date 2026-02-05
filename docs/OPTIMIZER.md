# 优化器详解

## 1. 与普通 PyTorch 项目的区别

在传统的深度学习（如 CNN, Transformer）中，网络结构在训练开始前就已定义完毕。
*   **参数数量是固定的**：Conv 层的权重 shape 是 `(C_out, C_in, K, K)`，训练全程不变。
*   **优化器初始化一次**：`optimizer = Adam(model.parameters(), lr=...)` 执行一次即可。

而在 **3D Gaussian Splatting** 中：
*   **参数数量是动态变化的**：为了精细化重建场景，算法会不断地 **分裂 (Split)** 大的高斯球，或 **克隆 (Clone)** 欠拟合区域的高斯球，同时 **剪枝 (Prune)** 透明或过大的高斯球。这意味着模型参数的 Tensor 长度（第一维度 `N`）在训练过程中会不断改变。
*   **优化器状态需要同步更新**：Adam 优化器内部维护了动量（Momentum, `exp_avg`）和方差（Variance, `exp_avg_sq`）状态。当高斯球数量改变时，如果我们简单地重新创建一个新的优化器，所有的历史动量信息都会丢失，导致训练震荡或发散。

## 2. 为什么封装 `GaussianOptimizer` 类？

如果不封装，直接在训练脚本中操作，代码将变得极其难以维护。`GaussianOptimizer` 的核心职责是将 **模型参数的变更** 与 **优化器内部状态的变更** 绑定在一起。

它主要解决了以下问题：

1.  **参数组管理**: 不同类型的高斯属性（位置、旋转、缩放、球谐系数）需要极其不同的学习率。例如，位置（Position）主要受梯度自适应控制，而颜色（SH）学习率较小。
2.  **动态 LR 调度**: 位置（XYZ）参数的学习率在训练过程中需要进行特定的指数衰减，封装后可以自动处理 `step()` 中的 LR 更新。
3.  **无缝的增删查改**: 提供统一接口，让外部模块（如 `Densifier`）可以在修改高斯点云的同时，安全地更新优化器内部状态。

## 3. 核心难点：如何在 PyTorch 中“动态”修改 Optimizer？

PyTorch 的 `torch.optim.Optimizer` 并没有原生提供“修改参数 Shape”的 API。一旦 `param_group` 被创建，它默认参数是静态的。为了实现动态性，我们需要直接操作优化器的底层字典 `optimizer.state`。

### 难点 A: 剪枝 (Pruning) - 如何保留历史？
当我们删除第 `i` 个高斯球时，不仅要从 `model.xyz` 中删除第 `i` 行，还要从 `optimizer.state[model.xyz]['exp_avg']` 中删除对应的第 `i` 行。

**解决方案 (`prune_optimizer`)**:
1.  接收一个布尔掩码 `mask`（True 表示保留）。
2.  遍历优化器中的每个参数组（xyz, rotation, scaling 等）。
3.  获取该参数对应的内部状态 `stored_state`。
4.  对状态张量进行切片操作：`stored_state["exp_avg"][mask]`。
5.  **替换参数**: 删除旧的 `nn.Parameter`，创建一个新的（更短的）`nn.Parameter` 并重新绑定给优化器。

### 难点 B: 增植 (Densification) - 新球的状态如何初始化？
当我们分裂一个高斯球产生新球时，新球的momentum应该如何设置？如果设为随机噪声，会破坏训练稳定性；如果设为 0，则意味着“冷启动”。3DGS 选择了初始化为 0（无历史惯性）。

**解决方案 (`cat_tensors_to_optimizer`)**:
1.  接收新生成的参数张量字典 `extension_tensor`。
2.  遍历参数组，找到对应的状态。
3.  构造全零的状态张量 `torch.zeros_like(extension_tensor)`。
4.  **拼接状态**: 使用 `torch.cat` 将旧状态和新零状态拼接。
5.  **拼接参数**: 使用 `torch.cat` 将旧参数和新参数拼接，并重新绑定。

## 4. 关键代码解析

### 初始化参数组
在 `__init__` 中，优化器会自动调用模型的 `get_param_groups` 方法。这实现了配置与代码的解耦，学习率在 `config.yaml` 中定义，在此处生效。

```python
# 每个属性有独立的 LR
self.param_groups = [
    {'params': [xyz], 'lr': 0.00016, 'name': 'xyz'},
    {'params': [features_dc], 'lr': 0.0025, 'name': 'f_dc'},
    ...
]
```

### 状态拼接实现 (Code Snippet)

这是 `core/optimizer.py` 中处理参数增加的核心逻辑（简化版）：

```python
def cat_tensors_to_optimizer(self, tensors_dict):
    for group in self.optimizer.param_groups:
        # 1. 获取需要追加的新数据
        extension_tensor = tensors_dict[group["name"]]
        
        # 2. 获取优化器当前缓存的动量状态 (Momentum State)
        stored_state = self.optimizer.state.get(group['params'][0], None)
        
        if stored_state is not None:
            # 3. 扩展状态：旧状态 + 新生成的零状态
            # 新加入的高斯点没有历史动量，所以用 zeros
            stored_state["exp_avg"] = torch.cat([
                stored_state["exp_avg"],
                torch.zeros_like(extension_tensor)
            ], dim=0)
            
            # ... 同理处理 exp_avg_sq ...
            
            # 4. 删除旧引用的 Key
            del self.optimizer.state[group['params'][0]]
            
            # 5. 更新 Parameter 本身 (拼接 tensor 并封装回 nn.Parameter)
            group["params"][0] = nn.Parameter(
                torch.cat([group["params"][0], extension_tensor], dim=0).requires_grad_(True)
            )
            
            # 6. 将状态挂载回新的 Parameter Key 上
            self.optimizer.state[group['params'][0]] = stored_state
```

## 5. 总结

`GaussianOptimizer` 是连接 **静态 PyTorch 框架** 与 **动态 3DGS 算法** 的桥梁。

*   **对上层 (Trainer)**: 它表现得像一个标准的优化器，提供 `step()`, `zero_grad()`。
*   **对侧层 (Densifier)**: 它暴露了 `prune` 和 `cat` 接口，允许在不破坏训练稳定性的前提下修改模型结构。
*   **对底层 (PyTorch)**: 它通过并不优雅但极其必要的实现方法，直接修改了 Adam 内部维护的参数列表和状态字典。

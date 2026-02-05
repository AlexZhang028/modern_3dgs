# FreeTimeGS 核心训练策略实现指南

本文档总结了 FreeTimeGS 论文中相对于标准 3DGS 引入的三个核心修改：**4D 正则化**、**周期性重定位**以及**速度学习率退火**。

## 1. 全局参数设置 (Overview)

| 参数 | 符号 | 建议值 | 说明 |
| :--- | :--- | :--- | :--- |
| 总迭代次数 | `max_iter` | **30,000** | 与标准 3DGS 一致 |
| 基础不透明度 | $\sigma$ | Learnable | 3DGS 中的 `opacity` 参数 |
| 时间不透明度 | $\sigma(t)$ | Derived | 由时间中心 $\mu_t$ 和持续时间 $s$ 计算得出 |

---

## 2. 4D 正则化 (4D Regularization)

**目的**：在训练早期抑制高斯体过高的不透明度，防止梯度消失导致的局部极小值（特别是在快速运动区域）。

### 公式
$$ \mathcal{L}_{reg}(t) = \frac{1}{N} \sum_{i=1}^{N} (\sigma_i \cdot \text{sg}[\sigma_i(t)]) $$

*   $\sigma_i$: 第 $i$ 个高斯体的基础不透明度（sigmoid 激活后）。
*   $\sigma_i(t)$: 第 $i$ 个高斯体在当前帧 $t$ 的时间不透明度。
*   $\text{sg}[\cdot]$: 停止梯度操作 (Stop-Gradient, PyTorch 中的 `.detach()`)。
    *   *注意：正则化仅更新基础不透明度 $\sigma$，不更新时间参数。*

### 实现细节
*   **权重 ($\lambda_{reg}$)**: `0.01` (1e-2)
*   **作用时机**: 每次迭代计算 Loss 时加入。
*   **代码逻辑 (PyTorch 伪代码)**:

```python
# forward 过程
# opacity: 基础不透明度 (N, 1)
# temporal_opacity: 当前时间 t 计算出的时间系数 (N, 1)

# Loss 计算
reg_loss_weight = 0.01
# 使用 detach() 实现 stop-gradient，防止优化时间参数
reg_loss = (opacity * temporal_opacity.detach()).mean() 

total_loss += reg_loss_weight * reg_loss
```

## 3. 周期性重定位 (Periodic Relocation)

**目的**：解决动态场景中高斯体分布不均的问题。将贡献度低（不透明度低）的高斯体移动到需要更多细节（采样得分高）的区域。

### 采样得分 (Sampling Score)
每个高斯体的得分 $s$ 定义为：
$$ s = \lambda_g \nabla_g + \lambda_o \sigma $$

*   $\nabla_g$: 高斯体的空间位置梯度平均模长 (View-space position gradients)。
*   $\sigma$: 高斯体的基础不透明度。

### 实现细节
*   **触发频率**: 每 **100** 次迭代执行一次 (`iteration % 100 == 0`)。
*   **权重参数**:
    *   $\lambda_g = 0.5$
    *   $\lambda_o = 0.5$
*   **操作逻辑**:
    1.  **筛选“供体”**: 找出不透明度 $\sigma$ 低于阈值（例如 `min_opacity`，通常沿用 3DGS 的 0.005 或 0.01）的高斯体。
    2.  **筛选“受体”**: 计算所有高斯体的得分 $s$，选出得分最高的 Top-K 个区域。
    3.  **重定位**: 将“供体”的位置（及可能的其他属性）移动到“受体”的位置（通常添加少量随机扰动），或者直接删除“供体”并在“受体”位置克隆新的高斯体。

---

## 4. 速度学习率退火 (Velocity LR Annealing)

**目的**：在训练早期允许大幅度运动（学习快速运动），后期微调运动细节（学习复杂微小运动）。

### 公式 (指数衰减)
$$ \lambda_t = \lambda_0^{1-\alpha} \cdot \lambda_1^{\alpha} $$
其中归一化时间进度 $\alpha = \frac{\text{current\_iter}}{\text{max\_iter}}$ （即文中符号 $t$）。

### 参数设置
由于论文声明使用与 3DGS 相同的优化器设置，以下参数推断自标准 3DGS 的位置（Position）学习率配置：

*   **调度器类型**: 指数衰减 (Exponential / Log-linear Interpolation)
*   **初始学习率 ($\lambda_0$)**: **1.6e-4**
*   **最终学习率 ($\lambda_1$)**: **1.6e-6**
*   **作用对象**: 仅针对高斯体的**速度参数 (Velocity, $v$)**。

### 代码逻辑 (PyTorch 伪代码)
```Python
    def get_velocity_learning_rate(step):
        # step: 当前迭代步数
        # max_steps: 30000
        
        alpha = step / max_steps
        initial_lr = 1.6e-4
        final_lr = 1.6e-6
        
        # 对数空间插值 (指数衰减)
        # 也就是: lr = initial_lr * (decay_factor ** alpha)
        # 其中 decay_factor = final_lr / initial_lr
        lr = (initial_lr ** (1 - alpha)) * (final_lr ** alpha)
        return lr
    
    # 在训练循环中更新优化器
    for step in range(max_steps):
        new_lr = get_velocity_learning_rate(step)
        for param_group in optimizer.param_groups:
            if param_group["name"] == "velocity":
                param_group["lr"] = new_lr
        
        # ... training step ...
```
---

## 5. 总结：训练流程图


1.  **训练循环 (0 -> 30k)**:
    *   **更新 LR**: 计算并更新 `velocity` 的学习率 (Step 4)。
    *   **前向传播**: 
        *   根据 $v$ 和 $t$ 更新位置 $\mu_x(t)$。
        *   渲染图像。
    *   **计算 Loss**:
        *   `L1_loss` + `SSIM_loss` + `LPIPS_loss`
        *   **+ `Reg_loss` (Step 2)**
    *   **反向传播**: `loss.backward()`
    *   **周期性控制**:
        *   `if step % 100 == 0`: 执行 **重定位 (Step 3)**。
        *   (标准 3DGS 的 Densification 如 Clone/Split 照常进行，通常在 500-15k 步之间)。
    *   **优化器步**: `optimizer.step()`
   
## 6. 实现步骤

- [x] 速度学习率衰减
- [x] 4D正则化
- [ ] 周期性重定向
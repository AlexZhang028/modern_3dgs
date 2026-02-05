# FreeTimeGS 训练参数配置手册

本文档汇总了复现 FreeTimeGS 论文所需的关键超参数设置。这些参数基于论文 **Section 4.1 Implementation Details** 及原版 3DGS 代码库的默认配置整理。

## 1. 优化器参数设置 (Optimizer Hyperparameters)

针对 FreeTimeGS 引入的三个新增参数，需在优化器中设置特定的学习率。标准 3DGS 参数（如 SH, Opacity, Rotation）保持原版设置不变。

| 参数名称 | 代码变量名 | 符号 | 学习率 (Learning Rate) | 说明 |
| :--- | :--- | :---: | :--- | :--- |
| **Time Center** | `_t` | $\mu_t$ | **1.0e-4** | 设置较小以防止时间漂移过大，保持时序稳定性。 |
| **Duration** | `_t_scale` | $s$ | **5.0e-3** | 与标准 3DGS 的空间 `scaling` 学习率保持一致。 |
| **Velocity** | `_motion` | $v$ | **1.6e-4** | 与标准 3DGS 的空间 `position` (`xyz`) 初始学习率一致。 |
| *Position* | `_xyz` | $\mu_x$ | *1.6e-4* | (参考) 从 1.6e-4 指数衰减至 1.6e-6。 |

---

## 2. 损失函数权重与调度 (Loss Weights & Scheduling)

FreeTimeGS 引入了 4D 正则化项，该项具有特定的时间衰减策略。

| 损失项 | 符号 | 初始权重 | 衰减策略 | 作用 |
| :--- | :--- | :--- | :--- | :--- |
| **4D Regularization** | $\lambda_{reg}$ | **1.0** | **前 3,000 iters 生效**<br>随后衰减为 0 或移除 | 防止训练初期高斯球产生过高的不透明度或过长的持续时间（“不透明墙”效应）。 |
| **D-SSIM** | $\lambda_{dssim}$ | **0.2** | 全程保持不变 | 结构相似性损失，与标准 3DGS 一致。 |
| **L1 Loss** | $\lambda_{L1}$ | **0.8** | 全程保持不变 | 图像像素级差异损失 ($1 - \lambda_{dssim}$)。 |

---

## 3. 训练流程控制 (Training Schedule)

包括迭代次数、自适应控制（分裂/克隆）以及新增的重定位机制。

| 流程控制项 | 参数名 | 推荐值 | 说明 |
| :--- | :--- | :--- | :--- |
| **总迭代次数** | `iterations` | **30,000** | 标准训练时长。 |
| **重定位间隔** | `relocation_interval` | **500** | **[新增]** 每 500 iters 执行一次 Relocation，将无效高斯球移至高误差区域。 |
| **分裂/克隆间隔** | `densify_interval` | **100** | 标准 3DGS 设置 (通常在 500-15k iter 之间生效)。 |
| **不透明度阈值** | `opacity_threshold` | **0.005** | 用于 Pruning (剪枝) 和 Relocation 的筛选阈值。 |

---

## 4. 初始化设置 (Initialization Settings)

在 `preprocess` 阶段或 `create_from_pcd` 阶段使用的数值。

| 初始化项 | 推荐值 | 物理含义与备注 |
| :--- | :--- | :--- |
| **初始点云数量** | **100,000** | 从 RoMA 重建结果中随机降采样至 10w 个点作为初始种子。 |
| **初始持续时间** | **0.001** | 对应归一化时间下的极短瞬间。 |
| **存储格式** | **-6.9** | 代码中存储为对数形式：$\ln(0.001) \approx -6.907$。 |
| **初始速度** | **KNN 计算值** | 基于相邻帧 3D 点云位移计算，若无对应则设为 0。 |

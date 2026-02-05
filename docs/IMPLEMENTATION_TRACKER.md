# FreeTimeGS 实现进度追踪

## 1. 概览
本文档用于追踪 FreeTimeGS 的实现进度，FreeTimeGS 是一种扩展了 3D Gaussian Splatting 的动态场景重建方法。
实现遵循 `FreeTimeGS_repr.md` 和 `freetimegs.md` 中的指南。

## 2. 参数定义 (已修正)
根据 `freetimegs.md`，该模型使用 4D 原语扩展了标准 3DGS。

| 概念 | PLY 属性名 | 模型属性 | 形状 | 描述 |
| :--- | :--- | :--- | :--- | :--- |
| **时间中心** | `t` | `_t` | [N, 1] | 归一化时间中心 $\mu_t$ |
| **持续时间** | `t_scale` | `_t_scale` | [N, 1] | 对数空间持续时间 $\ln(s)$ |
| **运动** | `motion_0`, `motion_1`, `motion_2` | `_motion` | [N, 3] | 平移速度 $v$ |

**关于旋转的说明**：保留标准 3DGS 的 `rotation` (四元数) 作为高斯球的静态方向。FreeTimeGS **不**包含旋转运动 (角速度)。运动纯粹是平移的 ($x(t) = x + v(t-t_0)$)。

## 3. 实现状态

### A. 数据处理与初始化 (tools/freetimegs_init.py)
- [x] 多视图空间重建 (RoMA + 三角测量)
- [x] 时间运动估计 (KNN)
- [x] **修复**: 输出 PLY 属性名称 (目前使用 `nx/ny/nz`, `time`, `duration`。需要 `motion_0..2`, `t`, `t_scale`)。
- [x] **修复**: 确保 `rotation` (rot_0..3) 初始化为单位四元数 (静态方向)。

### B. 核心模型 (core/gaussian_model.py)
- [x] `FreeTimeGaussianModel` 子类。
- [x] 参数注册 (`_t`, `_t_scale`, `_motion`)。
- [x] 激活函数 (`t_scale` 使用 Exp)。
- [x] PLY 加载/保存钩子。
- [x] `get_at_time` 变换逻辑。
- [x] 参数组 (优化器)。
- [x] **验证**: 确保未暗示或实现旋转运动。

### C. 渲染器 (core/renderer.py)
- [x] `render_temporal` 函数。
- [x] 基于时间的位置更新 (Eq. 1)。
- [x] 时间不透明度调制 (Eq. 4)。
- [x] 不可见高斯球的剔除。

### D. 训练 (train.py)
- [x] 数据加载
- [x] 渲染循环集成 (自动选择 render_temporal)。
- [x] 4D 正则化损失 (权重线性衰减)。
- [x] 周期性重定位 (基于 Error Map 采样与 Depth 反投影)。

## 4. 当前任务与修复
- 验证训练流程的收敛性。
- 微调超参数 (lambda_reg, relocation_interval) 以适应不同数据集。

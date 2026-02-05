# FreeTimeGS 训练核心机制与公式汇总

本文档整理了 FreeTimeGS 论文中用于动态场景重建的关键数学模型、正则化策略及优化机制。

## 1. 4D 高斯原语定义 (4D Gaussian Primitives)

FreeTimeGS 在传统 3D 高斯各向异性协方差的基础上，引入了显式的**时间**和**运动**属性。

### 1.1 核心参数集合
每个高斯原语 $G_i$ 包含以下可学习参数：
* **空间中心 (Position):** $\mu_x \in \mathbb{R}^3$
* **时间中心 (Time Center):** $\mu_t \in \mathbb{R}$
* **持续时间 (Duration):** $s \in \mathbb{R}$ (存储为 $\ln s$)
* **速度向量 (Velocity):** $v \in \mathbb{R}^3$
* *其他标准参数: 旋转 $q$, 缩放 $S$, 不透明度 $\alpha$, 球谐系数 $SH$*

---

## 2. 前向传播公式 (Forward Pass)

在渲染时刻 $t$ 时，高斯原语的状态通过以下公式动态计算：

### 2.1 显式线性运动 (Explicit Linear Motion)
高斯球在时刻 $t$ 的空间位置由其初始位置、速度和时间差决定：

$$
\mu_{x}(t) = \mu_{x} + v \cdot (t - \mu_{t})
$$

### 2.2 时间不透明度调制 (Temporal Opacity Modulation)
利用高斯衰减函数控制高斯球在时间轴上的可见性（即“淡入淡出”机制）。
* **持续时间激活：** $s_{actual} = \exp(t\_scale)$
* **时间权重计算：**

$$
\sigma(t) = \exp\left(-\frac{1}{2}\left(\frac{t - \mu_{t}}{s}\right)^{2}\right)
$$

### 2.3 最终渲染属性
输入到光栅化器 (Rasterizer) 的不透明度是原始不透明度与时间权重的乘积：

$$
\alpha(t) = \alpha_{raw} \cdot \sigma(t)
$$

---

## 3. 训练损失函数 (Training Loss)

总损失函数由图像重建损失和特定的 4D 正则化损失组成。

### 3.1 总体 Loss
$$
\mathcal{L} = (1 - \lambda_{dssim})\mathcal{L}_1 + \lambda_{dssim}\mathcal{L}_{D-SSIM} + \lambda_{reg}\mathcal{L}_{reg}
$$

### 3.2 4D 正则化 (4D Regularization)
**目的：** 防止高斯球在训练初期具有过大的不透明度或过长的持续时间，从而避免产生遮挡梯度的“不透明墙”，鼓励模型探索更精细的时间和空间位置。

**公式：**
$$
\mathcal{L}_{reg} = \frac{1}{N} \sum_{i=1}^{N} \sigma_i(t) \cdot \alpha_i
$$
* **机制：** 惩罚高斯球在当前渲染时刻的有效可见性。
* **策略：** 仅在训练的早期阶段（例如前 3000 次迭代）施加此约束，随后将 $\lambda_{reg}$ 衰减为 0。

---

## 4. 自适应控制策略 (Adaptive Control)

除了标准的克隆与分裂（Densification），FreeTimeGS 引入了**周期性重定位**来解决动态物体跟踪丢失的问题。

### 4.1 周期性重定位 (Periodic Relocation)
**触发条件：** 每隔固定的迭代次数（例如每 500 iter）执行一次。

**步骤 1：识别无效高斯球 (Pruning)**
筛选出对当前重建贡献极小的高斯球（可能是时间没对上，或位置偏离）：
$$
M_{prune} = \{ i \mid \alpha_i(t) < \epsilon_{opacity} \}
$$

**步骤 2：识别重建误差区域 (Error Map Sampling)**
计算当前视角的 L1 误差图 $E$，并从中采样像素坐标 $(u, v)$，这些区域代表运动重建质量差的地方：
$$
E = |I_{GT} - I_{Render}|
$$
$$
(u_{new}, v_{new}) \sim \text{Sample}(E)
$$

**步骤 3：参数重置 (Resetting)**
将无效高斯球“瞬移”到误差区域：
* **位置重置：** $\mu_{x}^{new} = \text{Unproject}(u_{new}, v_{new}, \text{Depth}_{render})$
* **时间重置：** $\mu_{t}^{new} = t_{current\_view}$
* **速度重置：** $v^{new} = 0$ (或微小随机值)
* **状态重置：** 稍微提升不透明度，防止立即被再次剔除。

---

## 5. 初始化机制 (4D Initialization)

利用 RoMA 和 KNN 为参数提供物理先验。

* **时间初始化：**
    $$\mu_t = \text{Timestamp of the frame}$$
* **速度初始化：**
    利用 KNN 寻找相邻帧 ($t$ 与 $t+1$) 3D 点云的对应关系：
    $$v = P_{t+1}^{nn} - P_{t}$$
    *(其中 $P_{t+1}^{nn}$ 是 $t+1$ 帧中距离 $P_t$ 最近的点)*
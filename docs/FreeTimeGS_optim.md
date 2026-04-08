# FreeTimeGS 训练流程终极优化方案

## 1. 核心病因诊断：4D 时间稀疏性 vs. 3DGS 粗暴重置

通过一系列多维度可视化（梯度图、时间窗 `t_scale`、时间中心 `t`、速度场 `velocity`），我们确认模型在**物理表征上已学得极其完美**（动静分离清晰、时序渐变平滑、速度场无污染）。当前唯一且致命的瓶颈在于：**“算力供给（高斯球数量）受限与重置机制误杀”**。

*   **诊断一：动态区域极度“缺人”**
    *   **证据**：梯度图显示背景全暗，动态物体极亮。
    *   **结论**：模型迫切需要在这片区域分裂出更多的高斯点，来拟合高频运动细节。
*   **诊断二：“短命”的动态点遭遇“大屠杀”**
    *   **证据**：`t_scale` 热力图显示动态点存在时间极短（紫色）；强行拉长 Densification 会导致高斯球数量和 PSNR 发生周期性雪崩。
    *   **结论（核心病灶）**：当系统触发 **Opacity Reset**（不透明度重置为 0.01）时，具有极短 `t_scale` 的动态点在随后的随机时间戳采样中，极大概率因为“未在活跃期”而获得 **0 梯度**。它们无法“回血”，并在 4D 正则化的持续打压下，于下一次 Pruning（剪枝）时被当作透明噪点**惨遭系统抹杀**。

---

## 2. 核心优化行动 (Action Items)

为了打破这个死循环，我们需要在代码中实施以下三个维度的机制修改：

### 行动一：彻底解绑 Densification 与 Opacity Reset（最优先）
**目的**：允许动态物体在后期持续分裂生长，同时保护它们不被全局重置误杀。
**方案**：新增配置项 `opacity_reset_until_iter`。早期清理背景垃圾，但在后期（如 15,000 步后）让重置机制提前退休，而保留 Densification 继续运行至 35,000 步。
```python
# 在 trainer.py 的 train_step 函数中，修改 Opacity Reset 的触发逻辑
# 确保 config 中新增了 opacity_reset_until_iter (如 15000)

if iteration < self.config.opacity_reset_until_iter:
    if iteration % self.config.opacity_reset_interval == 0 or \
       (self.config.white_background and iteration == self.config.densify_from_iter):
        self._reset_opacity()
```

### 行动二：引入运动引导的动态分裂 (Motion-Guided Densification)
**目的**：响应梯度图的呼唤，把系统的算力（高斯球分裂的名额）精准倾斜给快速运动的物体。
**方案**：在 `densify.py` 中，根据每个高斯点的速度（Velocity）模长，动态按比例降低其分裂的梯度门槛（最高可降 75%）。速度越快，越容易触发分裂补充细节。
```python
# 在 densify.py 的 densify_and_split / densify_and_clone 中
# 替换原有的 selected_pts_mask 筛选逻辑

# 1. 获取所有点当前的速度模长
velocities = self.model.get_motion  #[N, 3]
speed = torch.norm(velocities, dim=-1, keepdim=True) #[N, 1]

# 2. 根据场景实际情况，设置一个速度参考上限 (例如 2.0)
speed_factor = torch.clamp(speed / 2.0, 0.0, 1.0)

# 3. 动态降低分裂门槛：速度越快，越容易满足分裂条件 (最多降低 75% 的门槛)
dynamic_threshold = grad_threshold * (1.0 - 0.75 * speed_factor) 

# 4. 使用每个点专属的 dynamic_threshold 进行梯度筛选
selected_pts_mask = (torch.norm(grads, dim=-1, keepdim=True) >= dynamic_threshold).squeeze()
```

### 行动三：保持 Relocate 的“传帮带”策略 (Velocity Inheritance)
**目的**：确保通过重定位（Relocation）生成的新点，能够迅速跟上动态物体的节奏。
**方案**：在 `_relocate_gaussians` 中，让新生点继承目标区域“土著（Receptors）”的速度向量（附加微小扰动），避免从 0 开始加速导致的运动模糊。
```python
# 在 trainer.py 的 _relocate_gaussians 中
# 在执行 self.model.relocate(final_mask, new_xyz, timestamp) 之前添加：

# 获取受体 (receptors) 的速度
inherited_velocity = self.model._motion[receptor_indices]

# 加入微小扰动防止重合，并赋给新生的高斯点
noise = torch.randn_like(inherited_velocity) * 0.05
self.model._motion[final_mask] = inherited_velocity + noise
```

---

## 3. 训练超参数建议一览表

在实施上述代码修改后，建议采用以下超参数组合进行最终训练（以总步数 50k 为例）：

| 参数名 | 推荐值 | 设定理由 |
| :--- | :--- | :--- |
| `densify_until_iter` | **35,000** | 给动态物体充足的时间去分裂和完善细节 |
| `opacity_reset_until_iter`| **15,000** | 早期清理背景垃圾，后期绝对保护动态点免遭误杀 |

---

## 4. 预期效果 (Expected Outcomes)

实施这套终极方案后，我们预期在 TensorBoard 和渲染结果中观察到：
1.  **Num Gaussians 曲线**：在 15k 步（Opacity Reset 停止）之后，数量会**稳步攀升**，并稳定在一个较高的数值（如 2M~3M），彻底告别断崖式暴跌。
2.  **PSNR 曲线**：告别锯齿状跳水，呈现出一条**极其平滑且斜率向上的健康收敛曲线**。
3.  **渲染画面**：随着后期高斯点在动态区域的爆炸式堆积，运动物体（大提琴、狗、人物）的边缘将从“一团红雾”逐渐剥离出**锐利的纹理和清晰的轮廓**。
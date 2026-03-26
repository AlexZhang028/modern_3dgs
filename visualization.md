### 1. 速度场可视化 (Velocity Map) —— 诊断“为什么模糊”

**目的**：判断模型是真的学会了“运动”，还是在用“静止的云”来欺骗 Loss。
**逻辑**：如果物体在动，但该区域渲染出的速度图是黑色的（速度为0），或者颜色杂乱无章（方向随机），说明模型根本没学对运动。

**实现代码 (Renderer)**：
在 rasterizer 拿到 Gaussian 属性后，不要渲染 RGB，而是渲染把速度映射后的颜色。

```python
# 在 render 函数中，或者单独写一个 render_velocity
# 假设 gaussians.get_motion 返回 [N, 3]

velocity = gaussians.get_motion
# 计算速度模长 (Speed)
speed = torch.norm(velocity, dim=1, keepdim=True)

# 归一化用于可视化 (根据你的场景尺度调整 max_val，例如 5.0)
# 颜色越亮，速度越快
velocity_norm = torch.clamp(speed / 5.0, 0.0, 1.0) 

# 或者：方向可视化 (RGB对应XYZ方向)
# 将 [-v, v] 映射到 [0, 1]
direction_color = (torch.nn.functional.normalize(velocity, dim=1) + 1.0) / 2.0

# 混合：用速度模长控制亮度，用方向控制色相
vis_color = direction_color * velocity_norm

# 然后把这个 vis_color 传给 rasterizer 替代 sh_features
# 注意：Rasterizer 还是需要 view-dependent 投影吗？不需要，这当作固有色渲染
# 你可能需要把 SH degree 设为 0
```

**如何解读**：
*   **全黑/极暗**：**核心问题**。模型没学到速度，全靠静态拟合。$\rightarrow$ 需要更激进的 Velocity LR。
*   **颜色杂乱（像噪点）**：速度方向是随机的。$\rightarrow$ 正则化不够，或者初始化太差。
*   **平滑的色块**：**正常**。例如向左跑的人应该是纯红色的，向上跳的球是绿色的。

---

### 2. 时间窗热力图 (Duration/Scale-t Heatmap) —— 诊断“为什么有拖影”

**目的**：判断拖影是因为 `t_scale` 太大（长曝光），还是因为点的数量太多（重叠）。
**逻辑**：将 `t_scale` 渲染成热力图。

**实现代码**：
```python
# 获取 t_scale (log space)
t_scale_log = gaussians._t_scale
# 转为直观的秒/归一化单位
duration = torch.exp(t_scale_log)

# 归一化可视化: 假设我们希望 duration < 0.05 (非常短)
# 红色 = 长时间存在 (坏事/背景)，蓝色 = 瞬间存在 (好事/动态)
heatmap = torch.clamp(duration / 0.5, 0.0, 1.0)

# 转为伪彩色 (简单起见用红通道)
vis_color = torch.cat([heatmap, torch.zeros_like(heatmap), 1.0 - heatmap], dim=1)
```

**如何解读**：
*   **动态物体呈红色**：**病灶确诊**。大提琴是红色的，说明它在时间轴上太“胖”了。$\rightarrow$ 加大 4D 正则化，强迫它变蓝。
*   **动态物体呈蓝色但依然模糊**：说明时间窗是对的，模糊是因为**点不够密**或**位置不准**。

---

### 3. 时间中心可视化 (Time-Center Rainbow) —— 诊断“时序错乱”

**目的**：检查高斯体是否正确地按照时间顺序排列。
**逻辑**：把 `mu_t`（高斯体的中心时间）映射为颜色。

**实现代码**：
```python
mu_t = gaussians.get_t # [N, 1], range 0~1

# 使用 matplotlib 的 colormap (如 turbo) 映射成 RGB
import matplotlib.cm as cm
# 注意：这通常需要在 CPU 上做，或者手写一个简单的 RGB 渐变
# 简单实现：R=t, G=1-t, B=0
vis_color = torch.cat([mu_t, 1.0 - mu_t, torch.zeros_like(mu_t)], dim=1)
```

**如何解读**：
*   **彩虹色轨迹**：**完美**。如果你看到一个运动物体的轨迹呈现“红->黄->绿->蓝”的渐变，说明模型用不同时间的高斯体接力完成了运动。
*   **单一颜色混杂**：**错误**。如果大提琴的轨迹全是“紫色”，说明模型在用同一批时间的高斯体通过移动来拟合，这通常是 FreeTimeGS 想要的效果。
*   **杂色噪点**：说明 `t` 的初始化或优化完全乱了。

---

### 4. 空间尺度热力图 (Spatial Scale Heatmap) —— 诊断“为什么没细节”

**目的**：判断模糊是因为“高斯球太大（分辨率不够）”。
**逻辑**：渲染高斯体 3D 尺度的平均值。

**实现代码**：
```python
scales = gaussians.get_scaling # [N, 3]
avg_scale = torch.mean(scales, dim=1, keepdim=True)

# 归一化：越小越好。红色=大球，蓝色=小球
# 阈值根据场景调整，比如 0.01
vis_color = torch.clamp(avg_scale / 0.01, 0.0, 1.0).repeat(1, 3)
```

**如何解读**：
*   **大提琴是白/红色的**：**高斯体太大了**。物理上无法画出锐利的边缘。$\rightarrow$ 降低 Densify 阈值，强制分裂。
*   **大提琴是黑/蓝色的**：高斯体已经很小了，但还是很糊。$\rightarrow$ 位置跑偏了，或者颜色没学对。

---

### 5. 梯度贡献图 (Gradient Contribution) —— 诊断“为什么不生长”

**目的**：判断为什么 Densification 没有在模糊区域发生。
**逻辑**：在训练步中，积累并渲染每个像素回传的 View-space Gradient。

**实现**：
这个稍微麻烦点，需要在 `backward` 之后，但在 `step` 之前。
利用 `viewspace_points.grad` 的模长，把它当作颜色属性，再调用一次 rasterizer（这一步不求导，只为了看图）。

**如何解读**：
*   **模糊区域是黑色的**：**梯度消失**。说明 Loss 认为这里已经很好了（可能是 L1 的妥协），或者 Culling 把它切掉了。
*   **模糊区域很亮**：**梯度很高**。说明模型知道这里错了，但 `densify_grad_threshold` 设置得太高，拦截了分裂请求。$\rightarrow$ 降低阈值。

---

### 建议执行步骤

不要一次性全做，建议按以下顺序开发 Debug 模式：

1.  **第一天：做 Velocity Map**。这是最关键的。
    *   *预期*：你会发现你的大提琴区域在 Velocity Map 上极其微弱，或者是杂乱的。
    *   *对策*：看到图后，你就敢大胆地把 `velocity_lr` 加上去了。

2.  **第二天：做 Duration (Scale-t) Map**。
    *   *预期*：你会发现拖影区域是“高亮”的。
    *   *对策*：看到图后，你就知道正则化是不是加得不够了。

**如何在代码中集成？**
建议在 `GaussianRenderer` 的 `forward` 中加一个参数 `render_mode`:

```python
def forward(self, viewpoint_camera, pc, ..., render_mode='rgb'):
    # ...
    if render_mode == 'rgb':
        shs = pc.get_features
    elif render_mode == 'velocity':
        # 替换 shs 为速度颜色
        shs = compute_velocity_color(pc)
    elif render_mode == 'duration':
        # 替换 shs 为时间颜色
        shs = compute_duration_color(pc)
    
    # Rasterize
    # ...
```
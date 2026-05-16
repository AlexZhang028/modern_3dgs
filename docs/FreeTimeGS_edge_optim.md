# FreeTimeGS 边缘虚影与运动模糊消除方案

本方案旨在解决训练后期动态物体（如快速挥动的手、篮球）边缘出现高频虚影和残影的问题。通过在底层物理逻辑上对**“空间尺度”**和**“时间尺度”**施加基于速度的压制，迫使运动区域的高斯球变得**“更细碎”**且**“更短命”**。

---

## 机制一：速度引导的空间尺度压制 (Velocity-Conditioned Scale Restriction)

**核心原理**：
在 `densify_and_split` （分裂）阶段，原本的逻辑是“高斯球大于固定体积才允许分裂”。现在我们引入动态阈值：**运动越快的高斯点，被允许的最大体积越小**。这强迫手和球边缘的大高斯球碎裂成极细小的针尖点，从而让运动边缘变得尖锐。

**目标文件**：`densify.py` 
**修改位置**：`GaussianDensifier` 类的 `densify_and_split` 方法中，找到判断 `scales` 的逻辑并替换。

```python
        # ---------- 修改 densify_and_split 中的尺度判断逻辑 ----------
        
        scales = self.model.get_scaling
        max_scales = torch.max(scales, dim=1).values  # [N]
        
        # 1. 获取默认的基础尺度阈值
        base_size_threshold = self.config.percent_dense * scene_extent
        
        # 2. 计算基于速度的动态尺度阈值
        if hasattr(self.model, 'get_motion') and self.model.get_motion is not None:
            # 获取速度并计算速度因子 (归一化到 0.0 ~ 1.0)
            velocities = self.model.get_motion
            speed = torch.nan_to_num(torch.norm(velocities, dim=-1), nan=0.0)
            speed_factor = torch.clamp(speed / 2.0, 0.0, 1.0)
            
            # 动态阈值：速度越快，允许的最大体积越小 (最多缩小到原本的 1/3)
            dynamic_size_threshold = base_size_threshold / (1.0 + 2.0 * speed_factor)
        else:
            dynamic_size_threshold = base_size_threshold
            
        # 3. 使用动态尺度阈值进行筛选
        # 只有梯度达标 且 尺寸大于动态阈值的高斯球才会被分裂
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            (max_scales > dynamic_size_threshold).to(device)
        )
```

---

## 机制二：速度惩罚时间窗 (Velocity-Aware Duration Penalty)

**核心原理**：
残影的本质是因为高斯球的存活时间太长。物理规律是：**速度越快的点，其生命周期必须越短**（类似于缩短相机快门时间以抓拍高速运动）。我们在 Loss 中直接惩罚“在单个生命周期内移动距离过长（$v \times t$）”的点，强迫高速运动点的 `t_scale` 急剧收缩。

**目标文件**：`trainer.py`
**修改位置**：`_compute_loss_hook` 方法的末尾，在 `return loss` 之前添加。

```python
        # ---------- 在 _compute_loss_hook 的末尾添加 (return loss 之前) ----------

        # --- 机制二：基于速度的时间窗压制 ---
        # 建议在中后期（如 15000 步之后）引入，防止早期干扰基础时空结构的建立
        if iteration > 15000 and hasattr(self.model, '_motion'):
            
            # 1. 获取速度模长
            velocities = self.model.get_motion
            # 【关键】必须 detach()，我们只根据速度去惩罚时间，绝对不能让 Loss 反向去减小速度
            speed = torch.norm(velocities, dim=-1).detach() # [N]
            
            # 2. 获取真实的持续时间 (exp 激活后，无需 detach，这是被优化的目标)
            duration = torch.exp(self.model._t_scale).squeeze() # [N]
            
            # 3. 设定惩罚权重 (可根据实际拖影严重程度微调，建议 0.01 ~ 0.05)
            lambda_motion_blur = 0.05 
            
            # 4. 仅对快速运动的点施加惩罚，保护静态背景
            # 假设速度大于 0.5 视为快速运动 (请根据你场景的实际速度范围调整)
            fast_mask = speed > 0.5 
            
            if fast_mask.any():
                # 惩罚项：速度 * 持续时间 (即该点在生命周期内划过的距离)
                motion_blur_loss = (speed[fast_mask] * duration[fast_mask]).mean()
                loss += lambda_motion_blur * motion_blur_loss
```
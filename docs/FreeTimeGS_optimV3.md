# FreeTimeGS 训练中后期高斯球数量波动优化方案

## 1. 现象分析：为什么会出现剧烈的锯齿状波动？

在引入了“温和衰减（Soft Decay）”以清理后期幽灵浮空点后，我们观察到高斯球总数（Num Gaussians）在 15,000 次迭代后出现了高达几十万级别的**锯齿状周期性剧烈波动**。

*   **病因解剖（The Simultaneous Trap）**：
    波动的本质在于 **“打折衰减”与“低值剪枝（Pruning）”在同一迭代步（或极近的迭代步）内同步触发**。
    1.  系统对静态点统一进行衰减（例如 `* 0.9`）。
    2.  大量原本有用的背景细节（其不透明度刚好在 `0.0055` 左右），瞬间跌破 `0.005` 的剪枝红线。
    3.  系统立即执行 Pruning，这些有用的点**还没来得及通过下一轮渲染获取正向梯度来“回血”**，就遭到了**大规模误杀**。
    4.  背景细节丢失导致局部 L1 Loss 瞬间飙升，产生巨大的分裂梯度。
    5.  模型在接下来的几百步内被迫“疯狂爆兵”弥补空缺，导致数量再度陡升，直到下一次误杀，形成锯齿死循环。

---

## 2. 核心优化策略 (Action Items)

为了平滑收敛曲线并精准清理真正的浮空噪点，我们从三个维度对垃圾回收与生长机制进行精细化改造：

### 💡 策略一：靶向温和衰减 (Opacity-Conditioned Decay)
**目的**：保护坚固的真实背景，精准打击“幽灵浮空点”。
**方案**：真实的浮空点通常是半透明的（Opacity 介于 0.1~0.4 之间），而真实的墙壁地面非常实（Opacity > 0.8）。
*   在触发衰减时，增加条件判断：**仅对 `opacity < 0.5` 且 `duration > 0.5` 的高斯球执行衰减**。
*   同时，将衰减系数从 `0.9` 放缓至 **`0.95`**。真正的浮空点依然会因为持续 0 梯度而在多次打折后“慢性死亡”，但单次掉血量的减少极大地缓解了系统震荡。
```python
def _reset_opacity(self, iteration: int):
        current_opacity = self.model.get_opacity
        
        if iteration <= self.config.opacity_reset_until_iter:
            # 阶段一：硬重置 (0.01)
            target_opacity = torch.min(current_opacity, torch.ones_like(current_opacity) * 0.01)
        else:
            # 阶段二：靶向温和衰减 (Soft Decay)
            target_opacity = current_opacity.clone()
            
            # 1. 筛选静态点 (假设 duration > 0.5 为背景)
            duration = torch.exp(self.model._t_scale) 
            is_static_mask = (duration > 0.5) 
            
            # 2. 筛选半透明点 (保护 opacity 已经很高、很坚固的真实背景)
            is_semi_transparent = (current_opacity < 0.5).squeeze()
            
            # 3. 目标：既是静态的，又是半透明的 (浮空点特征)
            target_mask = is_static_mask & is_semi_transparent
            
            # 4. 温和打折：改为 0.95，减少单次掉血量，防止引发梯度剧变
            target_opacity[target_mask] = current_opacity[target_mask] * 0.95
            
        opacities_new = utils.inverse_sigmoid(target_opacity)
        optimizable_tensors = self.optimizer.replace_tensor_to_optimizer(opacities_new, "opacity")
        self.model._opacity = optimizable_tensors["opacity"]
```

### 💡 策略二：错峰执行机制 (Staggered Execution)
**目的**：给予被误伤的健康高斯点充足的“自证清白/回血”时间。
**方案**：在代码逻辑中硬性剥离 Decay 与 Pruning 的同步执行。
*   如果当前迭代步刚刚触发了 Opacity Decay，则**跳过或挂起本轮的 `densify_and_prune`**。
*   强迫模型再向前渲染 100 步。有用的点会在这 100 步内重新积累起巨大的 L1 梯度，瞬间将透明度拉回安全线以上；而真正的浮空噪点则毫无反应，在下一次 Pruning 时被精准击杀。
```python
# 在 trainer.py 准备调用 densify_and_prune 的地方

        # --- 策略：动态分裂门槛 (Annealing Threshold) ---
        # 假设 config.densify_grad_threshold_init = 0.0001
        # 假设 config.densify_grad_threshold_final = 0.0003
        if iteration > 15000:
            # 15k 到 35k 之间，门槛从 0.0001 线性/平滑升至 0.0003
            progress = min((iteration - 15000) / 20000.0, 1.0)
            current_grad_threshold = 0.0001 + progress * (0.0003 - 0.0001)
        else:
            current_grad_threshold = self.config.densify_grad_threshold

        # --- 策略：错峰执行 (Staggered Execution) ---
        # 检查当前是否恰好是 Opacity Reset/Decay 触发的轮次
        just_decayed = (iteration % self.config.opacity_reset_interval == 0)

        if iteration > self.config.densify_from_iter and iteration % self.config.densify_interval == 0:
            size_threshold = 20 if iteration > self.config.opacity_reset_interval else None
            
            # 只有在没有打折的轮次，才允许执行完整的 Densify & Prune
            if not just_decayed:
                self.densifier.densify_and_prune(
                    iteration=iteration,
                    max_grad=current_grad_threshold, # 使用动态门槛
                    min_opacity=self.config.prune_opacity_threshold,
                    extent=self.scene_extent,
                    max_screen_size=size_threshold
                )
            else:
                # 刚打完折的轮次，给高斯点 100 步的渲染时间去恢复梯度
                # 直接跳过本轮剪枝，或者只克隆不剪枝 (简单起见直接 pass)
```

### 💡 策略三：动态分裂门槛 (Annealing Densify Threshold)
**目的**：防止模型在训练后期过度敏感，陷入无休止的微小误差修补。
**方案**：在 15k 步到 35k 步的后期阶段，将分裂的梯度阈值（`densify_grad_threshold`）**从 0.0001 线性平滑升至 0.0003**。
*   随着基础结构的稳固，系统应当越来越“懒得”分裂。只有当遇到真正的高频缺失时才允许爆兵，从而彻底压平数量曲线的波动。

---

## 3. 预期效果 (Expected Outcomes)

组合应用上述三大策略后，训练过程将迎来质变：
1.  **彻底告别锯齿**：`Num Gaussians` 曲线将在 15k 步后展现出极其顺滑的微小弧度波动，平稳逼近容量上限。
2.  **背景零损伤**：靶向衰减完美保护了静态背景的高频细节，不会再出现“打折导致的背景模糊”。
3.  **前景深度正确**：伴随温水煮青蛙式的 `0.95` 衰减，漂浮在动态物体前方的“幽灵点”将被悄无声息地抹除，画面的 Z-order（深度关系）与视觉纯净度达到最佳状态。
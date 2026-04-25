# FreeTimeGS 训练后期“浮空点/前景遮挡”消除方案

## 1. 核心病因诊断：从“欠拟合”到“过拟合”的代价

通过可视化和数据统计，我们发现动态物体的细节已大幅改善，高斯球数量（Num Gaussians）稳定在 2.4M 左右，达到了极好的模型容量。但随之出现了**“背景高斯球遮挡动态前景（浮空点/Floaters）”**的问题。

*   **诊断一：垃圾回收机制（Garbage Collection）的缺失**
    *   **病因**：为了保护动态点，我们之前在 15,000 步后彻底禁用了 Opacity Reset。这导致模型失去了全局清理能力。
    *   **后果**：动态物体在快速穿过背景时，由于运动极快，模型会“就近借用”或临时分裂出一些背景点拉到前景来降低当前帧的 Loss。由于没有了 Reset 机制，这些用完即弃的“幽灵点（Floaters）”永远不会跌破剪枝阈值（0.005），从而作为半透明的杂质永久残留在了半空中，破坏了正确的深度关系（Z-order）。

---

## 2. 核心优化行动 (Action Items)

为了在“保护新生动态点”与“清理废弃浮空点”之间取得平衡，我们需要实施**“温和衰减”**与**“时间感知”**相融合的高阶垃圾回收策略。

### 行动一：引入“温和衰减” (Soft Opacity Decay) 替代硬重置
**目的**：摒弃“一刀切”降至 0.01 的做法，改用乘法衰减。
**方案**：在 15,000 步之后，恢复周期性的 Opacity 处理。但不再硬重置，而是将目标高斯球的透明度**打 9 折（乘以 0.9）**。
*   **有用的结构**：会在随后的几步迭代中，通过强大的渲染正梯度瞬间回血至 1.0。
*   **废弃的幽灵点**：因为缺乏多视角的持续观测梯度，会随着周期性的打折慢慢“流血致死”（0.9 $\rightarrow$ 0.81 $\rightarrow$ 0.72...），最终自然跌破 0.005 的剪枝下限被系统安全抹除。
```python
# 在 trainer.py 的 train_step 函数中

# 修改前的逻辑可能是：
# if iteration < self.config.opacity_reset_until_iter:
#     if iteration % self.config.opacity_reset_interval == 0: ...

# --- 修改为以下逻辑 ---
# 确保在整个 Densification 阶段（如 35000 步之前）都执行垃圾回收逻辑
if iteration <= self.config.densify_until_iter:
    if iteration % self.config.opacity_reset_interval == 0 or \
       (self.config.white_background and iteration == self.config.densify_from_iter):
        self._reset_opacity(iteration)  # 传入 iteration 参数
```

### 行动二：4D 专属的“时间感知”免死金牌 (Time-Aware Protection)
**目的**：绝对保障 `t_scale` 极小的动态物体不被误伤。
**方案**：在执行上述“温和衰减”时，加入时间维度的 Mask 筛选。读取每个高斯球的持续时间（`duration = exp(t_scale)`），**仅对持续时间较长的静态/背景点执行 0.9 的衰减惩罚**，完全放过那些生存周期极短、极脆弱的核心动态点。
```python
def _reset_opacity(self, iteration: int):
        """
        Modified Opacity Reset: 
        - Hard reset (0.01) in the early stage.
        - Time-aware soft decay (x0.9) in the late stage to clean up floaters.
        """
        current_opacity = self.model.get_opacity
        
        # 假设 config 中有 opacity_reset_until_iter (例如 15000)
        if iteration <= self.config.opacity_reset_until_iter:
            # 阶段一：原版粗暴重置，彻底清理初始化垃圾
            target_opacity = torch.min(current_opacity, torch.ones_like(current_opacity) * 0.01)
        else:
            # 阶段二：温和衰减 (Soft Decay) 与 时间感知 (Time-Aware)
            target_opacity = current_opacity.clone()
            
            # 提取时间窗大小 (t_scale 是对数存储，需 exp 还原)
            duration = torch.exp(self.model._t_scale) 
            
            # 定义"静态/背景点" (例如持续时间超过 0.5)
            # 注意：请根据你 t_scale 的实际物理单位(归一化或秒)调整这个阈值 0.5
            is_static_mask = (duration > 0.5) 
            
            # 仅对静态背景点或废弃点打 9 折，放过脆弱的、刚生成的短时动态点
            target_opacity[is_static_mask] = current_opacity[is_static_mask] * 0.9
            
        # 转换为 Logit 并更新到 Optimizer 和 Model
        opacities_new = utils.inverse_sigmoid(target_opacity)
        optimizable_tensors = self.optimizer.replace_tensor_to_optimizer(opacities_new, "opacity")
        self.model._opacity = optimizable_tensors["opacity"]
        
        print(f"Opacity Cleanup Done at iter {iteration}.")
```
# FreeTimeGS 代码修正指南

本文档详细列出了针对现有代码的修正方案，旨在解决与论文原文不符的逻辑错误（特别是关于 $t$ 的定义、正则化公式的实现以及重定位的优化器状态处理）。

---

## 1. 修正速度学习率退火 (Optimizer)

**文件**: `optimizer.py`

**问题**: 原文指出 *"where t goes from 0 to 1 during training"*。这意味着退火是基于**训练迭代进度**，而不是视频时间戳。使用视频时间戳会导致模型无法学习视频后半段的大幅度动作。

**修正代码**:

```python
    def update_learning_rate(self, iteration: int, timestamp: Optional[float] = None):
        """
        Update learning rates.
        Args:
            iteration: Current iteration number (1-based).
            timestamp: (Optional) Not used for velocity annealing in the correct implementation.
        """
        # 1. Update Position LR (Standard 3DGS logic)
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler(iteration)
                param_group['lr'] = lr
            
            # 2. Update Velocity LR (FreeTimeGS specific)
            if param_group["name"] == "velocity":
                # 修正：使用训练进度作为 t (0.0 -> 1.0)
                # Config 中需要包含总迭代次数，通常是 30,000
                max_steps = self.config.iterations 
                train_progress_t = min(iteration / max_steps, 1.0)
                
                # 使用训练进度 t 计算指数衰减
                lr = self.velocity_scheduler(train_progress_t)
                param_group['lr'] = lr
```

---

## 2. 修正 4D 正则化公式 (Trainer)

**文件**: `trainer.py`

**问题**: 论文公式为 $\mathcal{L}_{reg}(t) = \frac{1}{N}\sum (\sigma * \text{sg}[\sigma(t)])$。
其中 $\sigma$ 是基础不透明度，$\sigma(t)$ 是时间权重（Eq.4，范围 0~1）。之前的代码错误地使用了渲染后的不透明度（即 $\sigma * \sigma(t)$）参与计算，导致了物理含义偏差。

**前置要求**: 确保你的 `renderer` 返回字典中包含 `base_opacity` (即 $\sigma$) 和 `temporal_weight` (即 $\sigma(t)$)。

**修正代码**:

```python
    def _compute_loss_hook(self, loss: torch.Tensor, rendered: Dict, iteration: int) -> torch.Tensor:
        # 4D Regularization
        # Formula: sum( base_opacity * sg[temporal_weight] )
        
        # 1. 获取基础不透明度 σ
        # 注意：这里需要是经过 Culling 后参与渲染的那部分 Gaussians 的基础参数
        base_opacity = rendered['base_opacity'] 
        
        # 2. 获取时间权重 σ(t) (Eq.4 的结果)
        # 这是一个 0.0 到 1.0 之间的系数，表示该高斯体在当前时间是否"活跃"
        temporal_weight = rendered['temporal_weight']
        
        # 3. 计算 Loss
        reg_weight = 0.01  # Lambda_reg
        
        # 关键修正：Detach 时间权重，只惩罚基础不透明度
        # 物理含义：如果一个高斯体在当前时间 t 是活跃的(weight高)，
        # 我们就抑制它的基础不透明度，防止它在早期形成遮挡墙。
        l_reg = (base_opacity * temporal_weight.detach()).mean()
        
        loss += reg_weight * l_reg
            
        return loss
```

---

## 3. 新增重定位优化器重置 (Optimizer & Trainer)

**文件**: `optimizer.py` 和 `trainer.py`

**问题**: 当 `relocate` 将高斯体移动到新位置时，Adam 优化器中仍然保存着旧位置的动量（Momentum, `exp_avg`）。如果不清除，下一次迭代时，旧的动量会将高斯体“弹飞”，导致重定位失效。

**步骤 1: 在 `optimizer.py` 添加重置方法**

```python
    def reset_optimizer_state(self, mask: torch.Tensor):
        """
        Reset Adam optimizer state (momentum) for specific Gaussian indices.
        Used after relocation or pruning.
        
        Args:
            mask: Boolean mask of Gaussians to reset [N]
        """
        for group in self.optimizer.param_groups:
            # 假设 group['params'][0] 的形状 dim 0 等于高斯体数量
            param = group["params"][0]
            if param.shape[0] != mask.shape[0]:
                continue
                
            state = self.optimizer.state.get(param, None)
            if state is not None:
                # 清零动量和二阶矩估计
                state["exp_avg"][mask] = 0.0
                state["exp_avg_sq"][mask] = 0.0
```

**步骤 2: 在 `trainer.py` 中调用**

```python
    def _relocate_gaussians(self, iteration: int, rendered: Dict, target: torch.Tensor, camera: Any):
        # ... (前面的逻辑：计算得分、筛选 mask、计算 new_xyz) ...

        # 执行模型层面的重定位
        self.model.relocate(prune_mask, new_xyz, timestamp)
        
        # --- 新增 ---
        # 重置优化器状态，让这些点"重新开始做人"
        self.optimizer.reset_optimizer_state(prune_mask)
        
        print(f"[Relocation] Relocated {num_prune} gaussians at iter {iteration}")
```

---

## 4. 调整初始化参数 (Gaussian Model)

**文件**: `gaussian_model.py`

**问题**: 原代码将 `t_scale` 初始化为 `log(0.001)`。数值过小会导致高斯体在时间轴上极窄，在正则化介入前可能就因为不可见而被梯度忽略或被剪枝。

**修正代码**:

```python
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float = 1.0):
        # ... (前文代码) ...

        # Initialize t_scale (Duration)
        if pcd.t_scale is not None:
             self._gaussian_params['t_scale'] = nn.Parameter(
                 torch.from_numpy(pcd.t_scale.copy()).float().to(self.device), requires_grad=True
             )
        else:
             # 建议修改：使用 log(1.0) = 0.0 或 log(0.5)
             # 让高斯体初始时在时间轴上覆盖较广，然后依靠 4D Regularization 将其"削瘦"
             self._gaussian_params['t_scale'] = nn.Parameter(
                 torch.zeros((num_points, 1), device=self.device), requires_grad=True
             )
```

---

## 5. 检查列表 (Checklist)

在运行代码前，请确认：

1.  [ ] **配置**: `config.iterations` 是否设置为 30,000？
2.  [ ] **渲染器**: `renderer` 的输出是否包含了 `temporal_weight` 和 `base_opacity`？
3.  [ ] **调度器**: 确认 `config.velocity_lr` 约为 `1.6e-4` (最终衰减至 `1.6e-6`)。
4.  [ ] **频率**: `config.relocation_interval` 是否默认为 100？

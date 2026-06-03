# 动静解耦优化方案 (Phase-Based Decoupled Training)

## 1. 问题根因澄清

### 现象

动态加权方案（`FreeTimeGS_dynamic_optim.md`）在实践中会导致静态背景质量下降。

### 误区：不是梯度放大的问题

直觉上会认为"动态像素 Loss 放大 → 背景高斯球也被波及"。但 `dynamic_weight_map` 的像素值
本身就是由短 `t_scale` 的动态高斯球渲染出来的；背景高斯球在这些像素里几乎被动态物体遮挡，
贡献权重接近零，实际接收到的额外梯度放大并不显著。

### 真正的耦合机制：优化资源竞争

1. **密化预算被动态区域抢占**
   动态像素 Loss 更高 → 动态高斯梯度范数更大 → `densify_and_clone/split` 更多在动态区域
   触发 → 高斯球总数（受显存限制）中，动态区域持续扩张 → 静态区域密化压力相对不足 →
   静态质量停滞甚至退化。

2. **Adam 优化关注度的此消彼长**
   总迭代次数固定，更多的 loss 信号来自动态区域，Adam 的有效学习率向动态区域倾斜，
   静态区域的参数更新量相对减少。

### 结论

这是**资源竞争**，不是梯度幅值问题。解法不能只在梯度幅值上做文章，必须从源头把两类高斯球的
优化路径物理隔离。

---

## 2. 方案：分阶段梯度路由（单模型，无需合成）

核心思路：**同一个模型，通过训练阶段切换来交替独立优化静态和动态高斯球**。

每个 training step 结束 `backward()` 之后、`optimizer.step()` 之前，判断当前所处阶段，
然后将"本阶段不应更新"的那类高斯球的参数梯度清零。密化统计也同步屏蔽。

```
Phase 1 (0 ~ joint_end, 如 0-15k)：联合训练
  └─ 正常更新全部高斯球，建立基础动静分离

Phase 2 (joint_end ~ dynamic_end, 如 15k-30k)：动态专注
  └─ loss 保持 dynamic boost；backward 后清零静态高斯球的参数梯度
  └─ 同步清零静态高斯球在 densifier 中的梯度统计（防止静态区域被触发密化）

Phase 3 (dynamic_end ~ end, 如 30k-50k)：静态专注
  └─ 关闭 dynamic boost；backward 后清零动态高斯球的参数梯度
  └─ 同步清零动态高斯球的密化统计
```

---

## 3. 实现代码

### Step 1：在 TrainerConfig 中添加阶段配置（config/config.py）

```python
# 分阶段解耦训练
decouple_training_enabled: bool = False
decouple_joint_end_iter: int = 15000   # Phase 1 结束，进入动态专注
decouple_dynamic_end_iter: int = 30000 # Phase 2 结束，进入静态专注
decouple_static_duration_threshold: float = 0.5  # exp(t_scale) > 此值视为静态
```

### Step 2：在 train_step 的 backward 之后插入梯度路由逻辑

在 `loss.backward()` 之后、`# 8. Adaptive Control` 的 `with torch.no_grad():` 块之前添加：

```python
        # ---- 分阶段梯度路由 ----
        decouple_enabled = getattr(self.config, 'decouple_training_enabled', False)
        if decouple_enabled and hasattr(self.model, '_t_scale') \
                and self.model._t_scale is not None:

            joint_end   = getattr(self.config, 'decouple_joint_end_iter',   15000)
            dynamic_end = getattr(self.config, 'decouple_dynamic_end_iter', 30000)
            dur_thresh  = getattr(self.config, 'decouple_static_duration_threshold', 0.5)

            if iteration > joint_end:  # Phase 2 或 Phase 3
                with torch.no_grad():
                    # 基于 t_scale 实时分类（每步重新计算，参数在持续学习）
                    duration = self.model.get_t_scale.squeeze()  # [N]，exp 已激活
                    is_static  = duration > dur_thresh
                    is_dynamic = ~is_static

                    if iteration <= dynamic_end:
                        # Phase 2：动态专注 —— 清零静态高斯球参数梯度
                        freeze_mask = is_static
                        phase_name = "dynamic_focus"
                    else:
                        # Phase 3：静态专注 —— 清零动态高斯球参数梯度
                        freeze_mask = is_dynamic
                        phase_name = "static_focus"

                    if freeze_mask.any():
                        # 空间/颜色/几何参数全部冻结
                        frozen_params = ['xyz', 'features_dc', 'features_rest',
                                         'opacity', 'scaling', 'rotation']
                        for key in frozen_params:
                            param = self.model._gaussian_params.get(key)
                            if param is not None and param.grad is not None:
                                param.grad[freeze_mask] = 0.0

                        # 时序参数：动态专注阶段允许动态高斯球自由优化 t/t_scale/motion；
                        # 静态专注阶段冻结动态高斯球的时序参数（防止静态阶段扰乱已建立的时空结构）
                        if phase_name == "static_focus":
                            for key in ['t', 't_scale', 'motion']:
                                param = self.model._gaussian_params.get(key)
                                if param is not None and param.grad is not None:
                                    param.grad[freeze_mask] = 0.0

                        # 同步屏蔽密化统计，防止被冻结类别的高斯球触发密化/克隆
                        # （densifier 在本 step 的 update_stats 尚未调用，这里提前清零可等效于屏蔽）
                        # 注意：这里直接清零 densifier 中上一步残留的统计，
                        # 真正的本步屏蔽在 update_stats 之后的另一行
                        self.densifier.xyz_gradient_accum[freeze_mask] = 0.0
                        self.densifier.denom[freeze_mask] = 0.0
        # ---- 梯度路由结束 ----
```

### Step 3：update_stats 之后再次屏蔽密化统计

在 `self.densifier.update_stats(...)` 调用之后（约 line 583-587），紧接着添加：

```python
                    # ---- 分阶段密化屏蔽（与 Step 2 配合） ----
                    if decouple_enabled and hasattr(self.model, '_t_scale') \
                            and self.model._t_scale is not None \
                            and iteration > joint_end:
                        with torch.no_grad():
                            dur_thresh = getattr(self.config,
                                                 'decouple_static_duration_threshold', 0.5)
                            duration   = self.model.get_t_scale.squeeze()
                            is_static  = duration > dur_thresh
                            is_dynamic = ~is_static

                            if iteration <= dynamic_end:
                                self.densifier.xyz_gradient_accum[is_static] = 0.0
                                self.densifier.denom[is_static] = 0.0
                            else:
                                self.densifier.xyz_gradient_accum[is_dynamic] = 0.0
                                self.densifier.denom[is_dynamic] = 0.0
                    # ---- 密化屏蔽结束 ----
```

---

## 4. 与你原始方案（两次独立训练）的对比

你最初的想法：用预训练模型生成 segmentation mask → 分两次训练（一次静态、一次动态）。
这个方向是对的，但有以下困难：

| 问题 | 两次独立训练（原方案） | 分阶段梯度路由（本方案） |
|---|---|---|
| 推理复杂度 | 需要合成两个 PLY，边界有伪影 | 单一模型，推理无变化 |
| 训练成本 | 预训练 + 静态训练 + 动态训练 = 3 runs | 1 run，阶段内部切换 |
| Mask 精度依赖 | 强依赖（mask 错误 = 有缺口区域） | 弱依赖（t_scale 阈值软分类，每步更新）|
| 动静高斯球的交互 | 完全隔离（可能导致边界缺失） | 前期 (Phase 1) 共同学习边界，后期独立精化 |

本方案核心优势：**Phase 1 让两类高斯球有机会共同建立边界**，Phase 2/3 再独立精化，
避免了硬分割导致的边界缺口。

---

## 5. 调试建议

### 推荐的 YAML 配置段

```yaml
trainer:
  decouple_training_enabled: true
  decouple_joint_end_iter: 15000
  decouple_dynamic_end_iter: 30000
  decouple_static_duration_threshold: 0.5
  # dynamic_weighting 同时开启，Phase 2 阶段叠加 boost
  dynamic_weighting_enabled: true
  dynamic_boost_start_iter: 15000
  dynamic_boost_end_iter: 30000
  max_dynamic_boost: 10.0
```

### TensorBoard 监控

在 `_log_metrics` 中临时添加以下指标，确认解耦有效：

```python
if getattr(self.config, 'decouple_training_enabled', False) and self.writer:
    with torch.no_grad():
        dur = self.model.get_t_scale.squeeze()
        n_static  = (dur > getattr(self.config, 'decouple_static_duration_threshold', 0.5)).sum()
        n_dynamic = self.model.num_points - n_static
    self.writer.add_scalar('Decouple/n_static',  n_static.item(),  iteration)
    self.writer.add_scalar('Decouple/n_dynamic', n_dynamic.item(), iteration)
```

正常工作时，Phase 2 期间 `n_dynamic` 应持续增长（动态区域密化），`n_static` 相对稳定；
Phase 3 期间反之。

### 主要风险与对策

**风险**：Phase 2 结束时，由于静态高斯球 15k 步没有更新，若动态物体移动幅度大、
遮挡了大片背景，背景的静态高斯球在那片区域的位置可能过时。

**对策**：Phase 3 中对静态高斯球执行一次较激进的 opacity reset（强制清理浮影），
然后让 Phase 3 的静态专注训练重建这些区域。

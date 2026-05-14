## 动态物体加权（Dynamic Region Weighting）实现说明

本文档用中文总结当前仓库中关于“动态物体加权”功能的实现细节、关键配置项、实现位置与注意事项，便于调参与排查。

### 目标
- 在训练中对动态像素/点增加更高的重建损失权重，使模型在动态区域保留更多高频细节和点云密度。

### 配置项（位于 Trainer 配置）
- `dynamic_weighting_enabled`：开关（bool）。
- `dynamic_boost_start_iter` / `dynamic_boost_end_iter`：增益开始/结束迭代。
- `max_dynamic_boost`：最大增益倍数（B）。
- `dynamic_boost_curve_power`：控制增益曲线的幂（k），用于指数 easing。
- `dynamic_duration_static` / `dynamic_duration_dynamic`：用来把 `t_scale` 映射到 0..1 动态评分的时间窗参数。
- `dynamic_mask_threshold`：评估时把权重图二值化的阈值。

这些字段在 `config/config.py` / `config/parser.py` / 你的 YAML 实验文件中定义并读取。

### 核心实现位置
- 训练主流程：`core/trainer.py::train_step()` — 计算 `current_boost`，在 `current_boost>0` 时生成动态权重图并用于加权 L1。
- 评估/可视化：`core/trainer.py::_evaluate_set()` — 复用相同的 t_scale 映射，渲染动态权重热图并记录 masked PSNR（动态/静态分开）。
- Densification：`core/densify.py` — motion-guided threshold 与 clone/split/prune 的逻辑（在选择时使用速度因子降低或提高阈值）。
- 模型时间尺度：`core/gaussian_model.py` — `get_t_scale` 已经对内部 `_t_scale` 做了激活（exp），训练/评估应使用 `get_t_scale` 而不是对原始 `_t_scale` 再做 exp。

### 算法细节（训练时）
1. 计算增益 `current_boost`：
   - 在 `dynamic_boost_start_iter` 到 `dynamic_boost_end_iter` 之间做指数 easing：
     eased(p) = (e^{k p} - 1) / (e^{k} - 1), p ∈ [0,1], k = `dynamic_boost_curve_power`。
   - 最终 `current_boost = max_dynamic_boost * eased(p)`。

2. 计算每个高斯点/像素的动态得分：
   - 读取模型的激活时长：`duration = self.model.get_t_scale`（已 exp）。
   - dynamic_score = clamp((static_dur - duration) / denom, 0, 1)，其中 `denom = max(static_dur - dynamic_dur, 1e-6)`。
   - 把 `dynamic_score` 扩展为 `colors_override`（shape `[N,3]`）并传给 renderer，得到一张单通道的渲染权重图 `dynamic_weight_map`（范围 0..1）。

3. 形成像素级损失倍数并计算损失：
   - `pixel_loss_multiplier = 1.0 + current_boost * dynamic_weight_map.detach()`。
   - 用 `pixel_loss_multiplier` 乘以逐像素绝对误差（L1）来得到加权 L1；若 `pixel_loss_multiplier` 为标量则使用普通 L1。

4. 重要的实现与安全细节
   - Secondary render（权重图渲染）与任何 in-place 参数修改都运行在 `torch.no_grad()` 下，且用 `.detach()` 明确切断计算图，避免把像素级权重反传回 `t_scale` 或 `motion`（防止 autograd 泄漏）。
   - 在 `core/densify.py` 中，速度 `speed = norm(motion)` 使用 `torch.nan_to_num(speed, nan=0.0)` 以避免 NaN 污染门控阈值。
   - 对于 relocate / replace param 操作，使用 `optimizer.replace_tensor_to_optimizer` 与 `optimizer.reset_optimizer_state(mask)` 来重置或扩展 Adam 状态，避免动量残留导致学习率突变。

### 评估与可视化
- 在 `_evaluate_set()`：在 `iteration >= dynamic_boost_start_iter` 时，渲染并记录 `dynamic_weight_map` 的热图（用 `matplotlib` colormap colorize，fallback 为灰度），并把图写入 TensorBoard (`{prefix}_DynamicWeight/{camera.image_name}`)。
- 计算 Masked PSNR：使用 `dynamic_mask_threshold` 将 `dynamic_weight_map` 二值化生成 `dynamic_mask` 与 `static_mask`，分别计算 `masked_psnr_dynamic` 与 `masked_psnr_static` 并写到 TensorBoard。

### 监控指标（已记录到 TensorBoard）
- `DynamicWeight/boost`：当前 boost 大小。
- `DynamicWeight/map_mean` / `map_max`：权重图统计。
- `DynamicWeight/multiplier_mean` / `multiplier_max`：像素倍数统计。
- `params/t_scale_log`：`t_scale` 的直方图。
- `params/velocity_norm`：速度分布直方图。
- `Train/masked_psnr_dynamic` / `Train/masked_psnr_static`：评估集的分区 PSNR。
- 以及渲染图像：`Train_DynamicWeight/*` 热图。

### 性能/开销注意点
- `dynamic_weight_map` 的二次 render 会引入额外的 GPU 负载，尤其当同时开启 densify/relocate（参数增删）时，会造成 20k-30k 区间的 iteration_time 增大。原因：densify/relocate 操作会修改参数集合并触发 optimizer 状态操作（内存分配、tensor concat、reset），与额外渲染并发会显著增加延迟。

调优建议：
- 如果出现 20k-30k 的显著开销，可以：
  1. 把 `densify_until_iter` 提前（例如 25000）以停止 densify/relocate。
  2. 提高 `relocation_interval`（减少 relocate 频率）。
  3. 缩小 `max_dynamic_boost` 或延后 `dynamic_boost_start_iter`，减小短时内梯度突变。

### 常见问题与排查步骤
- 如果观察到权重图影响反常（loss 跳变）：检查 `Loss/l1_base` 与 `Loss/l1` 是否分离（已记录），若 `l1_base` 稳定而 `l1` 上升，说明只是目标尺度改变。
- 如果怀疑 autograd 泄漏：确认所有 secondary render 都在 `torch.no_grad()` 下并对 `dynamic_weight_map` 使用 `.detach()`。
- 如果发现 relocation 导致 LR 或动量异常：确认 `optimizer.reset_optimizer_state(mask)` 对被替换/新增索引已正确清零 `exp_avg`/`exp_avg_sq`，并且 `replace_tensor_to_optimizer` 的实现不会误改 step/bias 计数。

---
文件位置：[docs/DYNAMIC_WEIGHTING_CN.md](docs/DYNAMIC_WEIGHTING_CN.md)

如需我把其中某一节拓展成可运行的调参脚本（如批量修改 `dynamic_boost_curve_power` 并跑几次对比），我可以继续补充。 

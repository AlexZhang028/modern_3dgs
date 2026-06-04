# FreeTimeGS 优化实现报告

## 背景与问题定义

本项目基于 FreeTimeGS（CVPR 2025）实现动态场景重建。原始实现在 Corgi 等快速运动场景中存在以下瓶颈：

1. **动静耦合**：对动态区域施加更高的 Loss 权重时，静态背景质量同步下降，原因是两类高斯球在密化预算和优化器资源上相互竞争。
2. **动态物体高斯球密度不足**：通用密化阈值无法为高速运动区域提供足够的几何细节。
3. **重定位策略退化**：旧的 Partial Copy 策略导致背景区域累积雾状伪影（对应 FreeTimeGS++[1] 的 Secret 4）。
4. **训练稳定性差**：多次训练结果方差显著，单次报告不可靠（对应 FreeTimeGS++ 的 Secret 5）。
5. **动静分类不一致**：静态/动态判断分散在代码各处，基于启发式 t_scale 阈值，缺乏统一依据。

---

## 实现内容概览

| 模块 | 方案 | 对应问题 | 参考来源 |
|---|---|---|---|
| 分阶段梯度路由 | Phase-based Decoupled Training | 问题 1 | 自研 |
| 动态球专属密化 | Phase 2 Dynamic Densification Boost | 问题 2 | 自研 |
| MCMC 风格重定位 | MCMC-style Relocation | 问题 3 | FreeTimeGS++[1] Secret 4 |
| 仿射色彩校正 | Affine Color Correction | 问题 4 | FreeTimeGS++[1] Secret 5 |
| 门控时间边缘化 | Gated Marginalization | 问题 5 | FreeTimeGS++[1] Secret 1 |
| 统一分类接口 | Gate-First Classification Helpers | 问题 5 | 自研 |

---

## 方案一：分阶段解耦训练（Phase-based Decoupled Training）

### 动机

动态加权方案（对动态像素的 L1 Loss 乘以最高 10× 的系数）在后期会导致静态背景质量下降。根因分析：耦合不在于梯度幅值，而在于**优化资源竞争**——动态区域的高梯度范数抢占了密化预算，使背景密化压力不足。

### 方案设计

将训练分为三个阶段，通过 `backward()` 后的梯度清零实现物理隔离：

```
Phase 1 (0 → joint_end, 默认 15k iter)
   └ 联合训练，静态/动态高斯球自由竞争，建立基础动静分离

Phase 2 (joint_end → dynamic_end, 默认 15k–30k iter)
  └ 动态专注：清零静态高斯球的 xyz/color/scale/rotation 梯度
  └ 同步屏蔽静态高斯球的密化统计，防止触发 clone/split
  └ 动态加权（loss × boost）同步激活，不干扰静态（其梯度已清零）

Phase 3 (dynamic_end → end, 默认 30k–50k iter)
  └ 静态专注：清零动态高斯球的所有参数梯度（含 t/t_scale/motion）
  └ 同步屏蔽动态高斯球的密化统计
```

### 关键实现

**文件**：`core/trainer.py`，`config/config.py`

```python
# backward() 后，optimizer.step() 前
freeze_mask = is_static if iteration <= dynamic_end else is_dynamic
for key in ['xyz', 'features_dc', 'features_rest', 'opacity', 'scaling', 'rotation']:
    p = model._gaussian_params.get(key)
    if p is not None and p.grad is not None:
        p.grad[freeze_mask] = 0.0
# Phase 3 额外冻结时序参数
if iteration > dynamic_end:
    for key in ['t', 't_scale', 'motion']:
        ...
```

**新增配置项**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `decouple_training_enabled` | `false` | 是否启用 |
| `decouple_joint_end_iter` | `15000` | Phase 1 结束 |
| `decouple_dynamic_end_iter` | `30000` | Phase 2 结束 |
| `decouple_static_duration_threshold` | `0.5` | 静态判断阈值（t_scale fallback） |

---

## 方案二：Phase 2 动态高斯球专属密化增强

### 动机

Phase 2 期间静态高斯球的密化统计已被屏蔽，因此降低全局梯度阈值只会影响动态高斯球，不会在静态区域产生多余分裂。

### 方案设计

- **降低梯度阈值**：Phase 2 期间将 `densify_grad_threshold` 乘以 `decouple_dynamic_densify_mult`（默认 0.5），即从 0.0001 降至 0.00005，触发更激进的 clone/split。
- **提高分裂数量**：通过 `densify_and_prune` 的新参数 `n_split_override`，Phase 2 期间将每次分裂产生的子球数从 N=2 提高至 N=3（`decouple_dynamic_n_split`）。

### 关键实现

**文件**：`core/densify.py`（新增 `n_split_override` 参数），`core/trainer.py`

```python
effective_grad_threshold = current_grad_threshold
effective_n_split = None
if decouple_enabled and joint_end < iteration <= dynamic_end:
    effective_grad_threshold *= decouple_dynamic_densify_mult  # 0.5
    effective_n_split = decouple_dynamic_n_split               # 3
self.densifier.densify_and_prune(..., max_grad=effective_grad_threshold,
                                      n_split_override=effective_n_split)
```

---

## 方案三：MCMC 风格重定位（FreeTimeGS++, Secret 4）

### 动机

原有重定位策略属于 Partial Copy——继承 receptor 的位置和速度，但将 opacity 重置为固定值 0.01、t_scale 随机重置。FreeTimeGS++[1] 的受控实验（Table 4）表明，此策略会导致背景出现持续闪烁噪声，PSNR 在三种策略中最低。

### 方案设计

改为 **MCMC 全参数继承 + 归一化**：

1. **全参数继承**：donor 复制 receptor 的所有参数（xyz、features、rotation、t、t_scale、motion、gate），而非保留 donor 自身的过时颜色和几何。
2. **MCMC 归一化**：若 n 个 donor 被分配到同一个 receptor，每个 donor 的 opacity 和 scale 按 `log(n+1)` 减小，防止重叠导致的透明度过度累积（雾状伪影）。

### 核心公式

```
donor_opacity_logit = receptor_opacity_logit - log(n_total)
donor_log_scale     = receptor_log_scale     - log(n_total) / 3
```

其中 `n_total = n_donors_at_this_receptor + 1`（+1 是 receptor 自身）。

典型情况 n_total=2（一对一重定位）：donor opacity 约为 receptor 的一半，精确抵消双重覆盖。

---

## 方案四：仿射色彩校正（FreeTimeGS++, Secret 5）

### 动机

FreeTimeGS++[1] Table S3 显示，corgi 场景的多次训练 PSNR 标准差高达 1–2 dB，是所有场景中方差最大的。根本原因是照明不一致和相机曝光差异迫使高斯球学习光度补偿，产生不稳定的优化轨迹。

### 方案设计

为每个训练 camera（以 `camera.uid` 为键）维护独立的可学习仿射变换：

```
corrected = scale[3] * pred_img + bias[3]
```

- `scale` 初始化为 1，`bias` 初始化为 0（恒等变换）
- 正则化：`λ_cc * (||scale-1||² + ||bias||²)` 防止学习极端变换
- 独立 Adam optimizer，每步随高斯球参数一起更新
- 状态随 checkpoint 保存和恢复

### 实现位置

**文件**：`core/trainer.py`

```python
# Trainer.__init__
self.color_correction = nn.ModuleDict({
    str(cam.uid): nn.ParameterList([
        nn.Parameter(torch.ones(3,  device='cuda')),  # scale
        nn.Parameter(torch.zeros(3, device='cuda')),  # bias
    ]) for cam in dataset.cameras
})
self.cc_optimizer = torch.optim.Adam(self.color_correction.parameters(), lr=cc_lr)
```

---

## 方案五：门控时间边缘化（FreeTimeGS++, Secret 1）

### 动机

原始 FreeTimeGS 使用 t_scale 的 exp 激活值（持续时间 s）做时间透明度调制：

```
σ(t) = exp(-½((t - μ_t)/s)²)
```

s 趋向极大时接近常数 1（伪静态），但这只是 optimization 的隐式结果。FreeTimeGS++[1] 将其形式化为可学习的**门控**机制，使动静分离变为显式可微的学习目标。

### 方案设计

每个 FreeTimeGS 高斯球新增可学习参数 `gate_logit`，激活后得到门控值 `g = sigmoid(γ · gate_logit)`（γ=20）：

```
ϕ(t) = g + (1 - g) · exp(-½((Δt/s)²))
```

- `g → 1`：始终可见，表现为持久静态背景
- `g → 0`：退化为原始 FreeTimeGS 时间透明度（瞬态动态物体）
- 正则化：`λ_gate · g·(1-g).mean()` 推动门控值收敛到 0 或 1（双峰分布）

### 实现位置

**文件**：`core/gaussian_model.py`（新增 gate 参数、PLY 读写、optimizer 组），`core/trainer.py`（gate 正则化）

```python
# get_at_time() 中
transient_weight = torch.exp(-0.5 * (delta_t / (s + 1e-7)) ** 2)
if 'gate' in self._gaussian_params:
    g = self.get_gate.float()
    temporal_weight = g + (1.0 - g) * transient_weight
else:
    temporal_weight = transient_weight
```

---

## 方案六：统一分类接口（Gate-First Classification Helpers）

### 动机

在实现 Gated Marginalization 之前，静态/动态判断分散于 trainer.py 的 7 处，均使用 `t_scale > threshold` 的启发式逻辑。引入 gate 参数后需要统一替换，且后续迭代中任何对分类逻辑的改动都只需修改一处。

### 方案设计

在 `Trainer` 基类中添加两个 helper 方法：

```python
def _get_static_mask(self) -> BoolTensor[N]:
    """gate > 0.5 when available; else t_scale > threshold"""
    if 'gate' in model._gaussian_params:
        return model.get_gate.squeeze() > 0.5
    return model.get_t_scale.squeeze() > dur_thresh

def _get_dynamic_score(self) -> FloatTensor[N, 1]:
    """1 - gate when available; else t_scale-based formula"""
    if 'gate' in model._gaussian_params:
        return (1.0 - model.get_gate).detach()
    # ...t_scale clamped formula...
```

**替换覆盖范围**：

| 位置 | 用途 |
|---|---|
| 梯度路由（Phase 2/3） | freeze_mask 构造 |
| 密化统计屏蔽 | freeze_mask 构造 |
| Opacity soft-decay | is_static_mask（只衰减静态高斯球）|
| 动态加权 dynamic_score（训练） | 渲染动态权重图 |
| 评估 masked-PSNR | 动态/静态区域分别计算 PSNR |
| TensorBoard 热图（评估） | 动态权重可视化 |
| TensorBoard 计数日志 | Decouple/n_static + n_dynamic |

---

## 代码变更汇总

| 文件 | 变更类型 | 主要内容 |
|---|---|---|
| `config/config.py` | 新增字段 | ModelConfig: gated_marginalization, gate_sharpness；OptimConfig: gate_lr；TrainerConfig: 分阶段参数、CC 参数、gate 参数 |
| `core/gaussian_model.py` | 新增功能 | gate 参数（初始化、PLY 读写、optimizer 组）；get_at_time() 门控公式 |
| `core/densify.py` | 接口扩展 | densify_and_prune() 新增 n_split_override 参数 |
| `core/trainer.py` | 核心逻辑 | 分阶段梯度路由、密化屏蔽；动态密化增强；MCMC relocation；CC 初始化/应用/优化；gate 正则化；统一分类 helpers |
| `config/corgi_edge_optim.yaml` | 配置 | 启用全部新特性 |
| `docs/FreeTimeGS_gradient_decouple.md` | 文档 | 分阶段解耦设计文档 |

**Git 分支**：`feature/freetimegspp-improvements`（已 push 至 origin）

---

## 参考文献

[1] Lee, L.Y., Kim, S., Kim, Y., Kim, S., Park, J. *FreeTimeGS++: Secrets of Dynamic Gaussian Splatting and Their Principles*. arXiv:2605.03337v2, 2026.

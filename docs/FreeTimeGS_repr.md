# reeTimeGS 实现指南：从 Vanilla 3DGS 到 4D 动态重建

## 1. 核心数据结构变更 (GaussianModel)

在原版 GaussianModel 类中，除了原有的 position (xyz), rotation, scaling, opacity, sh 之外，需要新增 3组可学习参数。

### 1.1 新增参数定义

<div _ngcontent-ng-c2604817520="" not-end-of-paragraph="" class="table-content not-end-of-paragraph" jslog="275421;track:impression,attention" data-hveid="0" decode-data-ved="1" data-ved="0CAAQ3ecQahgKEwjjhoW1mL2SAxUAAAAAHQAAAAAQqwE"><table data-path-to-node="6"><thead><tr><td><span data-path-to-node="6,0,0,0">参数名</span></td><td><span data-path-to-node="6,0,1,0">符号</span></td><td><span data-path-to-node="6,0,2,0">形状</span></td><td><span data-path-to-node="6,0,3,0">物理含义</span></td><td><span data-path-to-node="6,0,4,0">激活函数/预处理</span></td></tr></thead><tbody><tr><td><span data-path-to-node="6,1,0,0"><b data-path-to-node="6,1,0,0" data-index-in-node="0"><code data-path-to-node="6,1,0,0" data-index-in-node="0">_t</code></b></span></td><td><span data-path-to-node="6,1,1,0"><span class="math-inline" data-math="\mu_t" data-index-in-node="0"><span class="katex"><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.625em; vertical-align: -0.1944em;"></span><span class="mord"><span class="mord mathnormal">μ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.2806em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight">t</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span></span></span></span></span></span></td><td><span data-path-to-node="6,1,2,0"><code data-path-to-node="6,1,2,0" data-index-in-node="0">[N, 1]</code></span></td><td><span data-path-to-node="6,1,3,0"><b data-path-to-node="6,1,3,0" data-index-in-node="0">时间中心</b>。高斯球存在感最强的归一化时刻。</span></td><td><span data-path-to-node="6,1,4,0">无 (Linear)</span></td></tr><tr><td><span data-path-to-node="6,2,0,0"><b data-path-to-node="6,2,0,0" data-index-in-node="0"><code data-path-to-node="6,2,0,0" data-index-in-node="0">_t_scale</code></b></span></td><td><span data-path-to-node="6,2,1,0"><span class="math-inline" data-math="s" data-index-in-node="0"><span class="katex"><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.4306em;"></span><span class="mord mathnormal">s</span></span></span></span></span></span></td><td><span data-path-to-node="6,2,2,0"><code data-path-to-node="6,2,2,0" data-index-in-node="0">[N, 1]</code></span></td><td><span data-path-to-node="6,2,3,0"><b data-path-to-node="6,2,3,0" data-index-in-node="0">持续时间</b>。控制高斯球在时间轴上的“存活窗口”。</span></td><td><span data-path-to-node="6,2,4,0"><b data-path-to-node="6,2,4,0" data-index-in-node="0">Exp</b> (存储为 <span class="math-inline" data-math="\ln s" data-index-in-node="9"><span class="katex"><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.6944em;"></span><span class="mop">ln</span><span class="mspace" style="margin-right: 0.1667em;"></span><span class="mord mathnormal">s</span></span></span></span></span>)</span></td></tr><tr><td><span data-path-to-node="6,3,0,0"><b data-path-to-node="6,3,0,0" data-index-in-node="0"><code data-path-to-node="6,3,0,0" data-index-in-node="0">_motion</code></b></span></td><td><span data-path-to-node="6,3,1,0"><span class="math-inline" data-math="v" data-index-in-node="0"><span class="katex"><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.4306em;"></span><span class="mord mathnormal" style="margin-right: 0.0359em;">v</span></span></span></span></span></span></td><td><span data-path-to-node="6,3,2,0"><code data-path-to-node="6,3,2,0" data-index-in-node="0">[N, 3]</code></span></td><td><span data-path-to-node="6,3,3,0"><b data-path-to-node="6,3,3,0" data-index-in-node="0">速度向量</b>。定义高斯球随时间的线性位移。</span></td><td><span data-path-to-node="6,3,4,0">无 (Linear)</span></td></tr></tbody></table></div>

### 1.2 优化器设置

在 setup_training_args 中，需要为这三个新参数添加对应的 learning_rate。

- 建议：
  - motion: 类似于 xyz 的学习率，但通常稍低。
  - t: 较低的学习率，避免时间漂移过大。
  - t_scale: 类似于 scaling 的学习率。
  
## 2. 渲染管线改造 (Forward Pass)

渲染逻辑的核心在于：在光栅化之前，根据当前视角的时间戳，动态修改高斯球的属性。

伪代码逻辑 (render.py)

```Python
def render(viewpoint_camera, gaussian_model, ...):
    # 1. 获取当前帧的时间 (归一化到 0~1)
    # 注意：viewpoint_camera 需要被修改以包含 timestamp 信息
    current_time = viewpoint_camera.time 

    # 2. 获取原始参数
    xyz = gaussian_model.get_xyz
    opacity = gaussian_model.get_opacity
    
    # === FreeTimeGS 核心变换 ===
    
    # A. 提取新参数
    mu_t = gaussian_model.get_t
    s_duration = torch.exp(gaussian_model.get_t_scale) # 必须 exp
    velocity = gaussian_model.get_motion

    # B. 计算时间差 (支持 Broadcasting)
    delta_t = current_time - mu_t

    # C. 显式运动更新 (Explicit Motion)
    # Eq. 1: mu(t) = mu + v * (t - mu_t)
    xyz_at_t = xyz + velocity * delta_t

    # D. 时间不透明度调制 (Temporal Opacity Modulation)
    # Eq. 4: Gaussian Decay
    # 加上 epsilon 防止除零
    dist_sq = (delta_t / (s_duration + 1e-7)) ** 2
    temporal_mod = torch.exp(-0.5 * dist_sq)
    
    # 调制后的不透明度
    opacity_at_t = opacity * temporal_mod

    # E. (可选优化) 剔除不可见的高斯球
    # 如果 opacity_at_t 极小，直接 mask 掉，不送入光栅化器
    mask = opacity_at_t > 0.001
    
    # === 调用标准光栅化器 ===
    
    # 注意：传入的是变换后的 xyz_at_t 和 opacity_at_t
    rendered_image = rasterizer(
        means3D = xyz_at_t[mask],
        opacities = opacity_at_t[mask],
        scales = gaussian_model.get_scaling[mask],
        rotations = gaussian_model.get_rotation[mask],
        ...
    )

    return rendered_image
```
## 3. 训练循环与特殊机制 (Training Loop)

FreeTimeGS 的训练不仅仅是梯度下降，它引入了 4D 正则化 和 重定位 (Relocation) 策略来保证收敛。

### 3.1 损失函数设计

总 Loss 包含三部分：L1 Loss, D-SSIM Loss, 以及新增的 4D 正则化 Loss。

- 4D Regularization Loss ($\mathcal{L}_{reg}$):
  - 目的：防止高斯球在训练初期变成“一堵墙”（Opacity 过高且 Duration 过长），导致无法优化内部结构。
  - 计算：惩罚所有高斯球的时间不透明度均值。
  - 权重衰减：只在训练初期（如前 3000 iter）使用，之后衰减为 0。
  
### 3.2 训练流程伪代码 (train.py)
```Python
def training(dataset, opt, pipe, ...):
    
    # 初始化 (使用 RoMA + KNN 生成的 PLY)
    gaussian_model.load_ply("init_point_cloud.ply")
    
    first_iter = 0
    
    for iteration in range(first_iter, opt.iterations):
        
        # 1. 随机采样一个视口 (包含图像 + 时间)
        viewpoint_cam = dataset.next()
        
        # 2. 渲染 (Forward)
        render_pkg = render(viewpoint_cam, gaussian_model, ...)
        image, viewspace_point_tensor, ... = render_pkg
        
        # 3. 计算基础 Loss
        gt_image = viewpoint_cam.original_image
        Ll1 = l1_loss(image, gt_image) 
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim(image, gt_image)
        
        # 4. === 新增：4D 正则化 Loss ===
        if iteration < opt.reg_end_iter:
            # 计算当前视角下所有高斯球的不透明度均值
            # 注意：这里的 opacity 是经过 temporal_mod 调制后的
            opacity_loss = torch.mean(render_pkg["opacity_at_t"])
            
            # 权重随时间衰减 (例如线性衰减)
            reg_weight = get_linear_decay_weight(iteration)
            loss += reg_weight * opacity_loss
            
        # 5. Backward & Optimizer Step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # 6. === 核心差异：自适应控制 (Densification & Relocation) ===
        if iteration in densification_intervals:
            
            # A. 标准分裂与克隆 (Standard Clone/Split)
            # 基于 viewspace_point_tensor 的梯度 (x, y 的梯度)
            gaussian_model.densify_and_prune(...)
            
            # B. === 新增：周期性重定位 (Periodic Relocation) ===
            # 每隔一定步数 (如 500 iters) 执行
            if iteration % opt.relocation_interval == 0:
                
                # a. 找出 "没用" 的高斯球
                # 标准：不透明度低于阈值 (threshold < 0.01)
                prune_mask = gaussian_model.get_opacity < 0.01
                
                # b. 找出 "误差大" 的区域 (Under-reconstructed regions)
                # 计算当前 batch 的 Error Map (L1 difference)
                error_map = torch.abs(image - gt_image).mean(dim=0)
                # 采样坐标：在 error_map 大的地方随机选点 (u, v)
                new_pixel_coords = sample_from_pdf(error_map, num_samples=prune_mask.sum())
                
                # c. 瞬移 (Teleport)
                # 将 prune_mask 选中的高斯球参数重置：
                # - 位置: 反投影 new_pixel_coords 到 3D 空间 (利用 Depth 或视线)
                # - 时间: 重置为当前 viewpoint_cam.time
                # - 速度: 重置为 0 或保留微小随机值
                # - Opacity: 稍微调高一点以免马上被 prune
                gaussian_model.relocate(prune_mask, new_pixel_coords, viewpoint_cam)
```
## 4. 关键实现细节与避坑指南

### 4.1 参数梯度流 (Gradient Flow)

- Motion 的梯度：来源于 $t$ 时刻的位置偏差。如果 $t=0$ 时位置对，但 $t=1$ 时位置偏了，梯度会通过 xyz_at_t = xyz + v * dt 传导回 v。
- t_scale 的梯度：来源于背景/前景的区分。背景需要在所有帧可见，梯度会把 t_scale 推向大值；快速运动物体只需在特定帧可见，梯度会把 t_scale 推向小值。

### 4.2 静态背景处理

- 现象：如果不加控制，背景可能会闪烁。
- 机制：exp 激活的 t_scale 允许产生极大的持续时间（如 $10^5$），这让 $\sigma(t) \approx 1.0$ 恒成立。初始化时，如果某些点来自静态区域（比如 KNN 发现它在多帧间没怎么动），可以将它们的 t_scale 初始值设大一点，或者 motion 设为 0。
  
### 4.3 显存优化

- FreeTimeGS 由于需要多帧匹配初始化，点数可能较多。
- 在 render() 函数中，尽早使用 mask 剔除 opacity_at_t 接近 0 的点，不要把它们送入 Rasterizer，这对提升训练速度和降低显存占用至关重要。


### 4.4 初始化加载

- 你需要重写 GaussianModel.create_from_pcd。
- 读取 PLY 时，不仅要读 x, y, z，还要解析 t, t_scale, motion，并将它们加载到 Optimizer 的参数组中。

## 5. 总结：复现工作量清单
1. 预处理脚本：编写 RoMA + KNN 脚本，生成带有 motion 和 t 的 init.ply。
2. 模型类修改：在 GaussianModel 中注册 _t, _t_scale, _motion 为 Parameter。
3. 渲染函数修改：插入公式 Eq.1 (位置更新) 和 Eq.4 (不透明度调制)。
4. 训练循环修改：加入 reg_loss。实现 relocation 逻辑（采样 Error Map 并重置高斯球）。
5. 数据加载器修改：确保 Camera 对象携带归一化的 time 属性。
# FreeTimeGS 初始化流水线设计文档 (Multi-View Support)

## 1. 目标与背景

本脚本的主要目的是替代传统 3DGS 中基于静态 COLMAP 点云的初始化方式。针对动态场景，我们需要生成一个包含时空信息 (4D) 的初始点云，为 FreeTimeGS 的三个核心新增参数（t, t_scale, motion）提供合理的初值。

核心输入：

- 有序的视频帧序列（Frame 0 to Frame N）。
- 每一帧包含多视角图像（Camera 0 to Camera M）。
- 已标定的相机参数（需在同一世界坐标系下）。

核心输出：

- 一个单一的 .ply 文件，包含所有时刻的聚合点云及其动态属性。

## 2. 算法核心路径 (Pipeline Overview)

整体流程分为三个阶段：空间重建（Spatial） $\rightarrow$ 时间关联（Temporal） $\rightarrow$ 全局聚合（Aggregation）。

### 阶段一：单帧内的多视角空间重建 (Spatial Reconstruction)

目的：利用 RoMA 提取当前时刻 $t$ 的 3D 几何结构。策略：采用“链式匹配 (Chain Matching)”策略处理多视角，并进行帧内聚合。伪代码逻辑：

```Python
Initialize global_point_cloud_list = []

For t from 0 to Total_Frames - 1:
    
    # 1. 准备当前帧的数据
    current_cameras = frames[t].cameras  # [Cam0, Cam1, Cam2...]
    current_images = frames[t].images
    
    frame_t_points = [] # 用于存放当前帧所有视角产生的点
    
    # 2. 定义匹配对 (链式策略: 0-1, 1-2, 2-3...)
    # 这种策略能最大程度覆盖物体表面
    camera_pairs = generate_chain_pairs(len(current_cameras))
    
    For (cam_i, cam_j) in camera_pairs:
        
        # A. RoMA 特征匹配 (2D Domain)
        # 获取图像 I_i 和 I_j 之间的致密匹配点
        matches_2d = RoMA_Match(current_images[cam_i], current_images[cam_j])
        
        # B. 三角测量 (2D -> 3D)
        # 关键：利用 Projection Matrix (P = K[R|T]) 恢复 3D 坐标
        # 必须确保所有相机的 Pose 都在统一的世界坐标系下
        points_3d_local = Triangulate(matches_2d, P_cam_i, P_cam_j)
        
        # C. 收集点云
        frame_t_points.append(points_3d_local)
        
    # 3. 帧内聚合 (Aggregation)
    # 将本帧所有视角恢复出的点云合并
    # FreeTimeGS 允许点云重叠，无需复杂的融合去重
    cloud_at_t = Concatenate(frame_t_points)
    
    # 4. (可选) 随机降采样
    # 防止点数过多导致后续计算爆炸，每帧保留约 1w-2w 个高质量点即可
    cloud_at_t = RandomDownsample(cloud_at_t, target_count=20000)
    
    # 5. 暂存结果
    global_point_cloud_list.append(cloud_at_t)
```

### 阶段二：跨帧的时间关联与速度估计 (Temporal Motion Estimation)

目的：计算每个点的 motion 向量 ($v$)，并将 t 归一化。策略：利用 KNN 在相邻帧的聚合点云之间寻找对应关系。

伪代码逻辑：

```Python
Initialize final_data_containers = []

For t from 0 to Total_Frames - 1:
    
    curr_cloud = global_point_cloud_list[t]
    
    # 1. 确定目标帧 (通常是下一帧)
    # 边界处理：如果是最后一帧，则目标帧设为上一帧，或者速度设为0
    if t < Total_Frames - 1:
        target_cloud = global_point_cloud_list[t + 1]
        direction = 1.0
    else:
        target_cloud = global_point_cloud_list[t - 1] # 回溯
        direction = -1.0 # 标记方向反转，或仅用于借用位置
        
    # 2. KNN 搜索 (3D Domain)
    # 在 target_cloud 中找到与 curr_cloud 每个点最近的邻居
    # 假设：帧率足够高，物体位移小于场景尺度
    nearest_neighbors_indices = KNN(query=curr_cloud, target=target_cloud, k=1)
    nearest_points = target_cloud[nearest_neighbors_indices]
    
    # 3. 计算速度向量 (Motion)
    # v = P_next - P_curr
    if t < Total_Frames - 1:
        velocity = nearest_points - curr_cloud
    else:
        velocity = np.zeros_like(curr_cloud) # 简单处理：最后一帧静止
        
    # 4. 计算归一化时间 (Time)
    # t 必须映射到 [0, 1] 区间
    normalized_t = t / (Total_Frames - 1)
    time_array = Fill(value=normalized_t, shape=len(curr_cloud))
    
    # 5. 封装数据
    frame_data = {
        'xyz': curr_cloud,
        'motion': velocity,
        't': time_array
    }
    final_data_containers.append(frame_data)
```

### 阶段三：参数初始化与文件输出 (Formatting)

目的：计算 scale 和 t_scale，并将数据打包为 FreeTimeGS 兼容的 .ply 格式。

伪代码逻辑：

```Python
# 1. 全局合并
# 将所有帧的数据堆叠成一个巨大的数组 (N_total, C)
All_XYZ    = Concatenate([d['xyz'] for d in final_data_containers])
All_Motion = Concatenate([d['motion'] for d in final_data_containers])
All_Time   = Concatenate([d['t'] for d in final_data_containers])

# 2. 初始化空间尺度 (Scale) - 标准 3DGS 操作
# 计算每个点到最近3个点的平均距离
dist_sq = SimpleKNN_DistCUDA2(All_XYZ)
spatial_scale = Sqrt(dist_sq)

# 3. 初始化时间尺度 (Duration/t_scale) - FreeTimeGS 特有
# 存储为对数形式。初值应较小，表示高斯球仅在当前帧附近有效
# 例如：exp(-5.0) ≈ 0.006，约等于归一化时间下的极短瞬间
t_scale_log = Fill(value=-5.0, shape=len(All_XYZ))

# 4. 构建 PLY 元素
vertices = ConstructPlyElement(
    positions = All_XYZ,          # x, y, z
    normals   = All_Motion,       # 借用 normals 字段或自定义字段存储 motion_0,1,2
    time      = All_Time,         # t
    duration  = t_scale_log,      # t_scale
    scales    = Log(spatial_scale)# scale_0,1,2 (Log space)
    ...                           # 其他标准字段 (RGB, SH, Opacity...)
)

# 5. 保存
SaveToPly(vertices, "init_point_cloud.ply")
``` 
## 3. 关键技术细节备忘

### A. 相机对的选择 (Camera Pairing)

在多视角设置中，不要尝试匹配所有可能的组合（$C_N^2$）。

- 推荐：按物理位置顺序匹配 $(i, i+1)$。
- 补充：如果首尾相机重叠（环形阵列），增加 $(N-1, 0)$ 匹配对。
  
### B. 坐标系一致性 (Coordinate Consistency)

这是最容易出错的地方。
- 所有相机的 Projection Matrix 必须基于同一个 World Coordinate System。
- 如果每一帧的相机 Pose 是独立估计的（如分别跑了 COLMAP），则必须先进行轨迹对齐 (Trajectory Alignment)，否则计算出的 motion 会包含相机自身的运动漂移，导致训练失败。

### C. 显存与计算量控制

- RoMA 采样：RoMA 输出的点非常多，务必在三角测量后进行降采样。
- KNN 优化：如果总点数超过百万级，使用 scikit-learn 的 KDTree 或 BallTree 进行查询，或者使用 GPU 加速的 KNN 库（如 faiss 或 pytorch3d）。
   
### D. 边界条件处理

- 第一帧与最后一帧：
  - 第一帧的 motion 由 $P_1 - P_0$ 决定。
  - 最后一帧没有“下一帧”，通常将其 motion 设为 0，或者沿用上一帧的速度。在 FreeTimeGS 的优化过程中，这些初值会被梯度自动修正。

## 4. RoMa示例代码

根据RoMa官方的github repo，有以下示例代码

```Python
from PIL import Image
import torch
import cv2
from romatch import roma_outdoor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--im_A_path", default="assets/sacre_coeur_A.jpg", type=str)
    parser.add_argument("--im_B_path", default="assets/sacre_coeur_B.jpg", type=str)

    args, _ = parser.parse_known_args()
    im1_path = args.im_A_path
    im2_path = args.im_B_path

    # Create model
    roma_model = roma_outdoor(device=device)


    W_A, H_A = Image.open(im1_path).size
    W_B, H_B = Image.open(im2_path).size

    # Match
    warp, certainty = roma_model.match(im1_path, im2_path, device=device)
    # Sample matches for estimation
    matches, certainty = roma_model.sample(warp, certainty)
    kpts1, kpts2 = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)    
    F, mask = cv2.findFundamentalMat(
        kpts1.cpu().numpy(), kpts2.cpu().numpy(), ransacReprojThreshold=0.2, method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=10000
    )
```
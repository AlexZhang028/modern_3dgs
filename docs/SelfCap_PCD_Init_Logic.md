# 基于单帧点云输入的 FreeTimeGS 初始化脚本逻辑说明

对应脚本：`tools/process_selfcap_pcd.py`

## 1. 脚本目标

该脚本把一组按时间排序的单帧点云（`pcds/*.ply`）转换为 FreeTimeGS 可直接加载的初始化 PLY，核心是给每个点提供：

- 空间参数：`x,y,z`、`scale_0/1/2`、`rot_0/1/2/3`、`opacity`
- 颜色参数：`f_dc_*` 与 `f_rest_*`（SH 系数）
- 时空参数：`t`、`t_scale`、`motion_0/1/2`

换句话说，它不是简单拼接点云，而是构造一个带时间与速度先验的 4D 高斯初值。

## 2. 输入与输出

输入：

- 数据根目录（`--source_path`），默认读取子目录 `pcds/`
- 多帧 PLY（每帧包含至少 `x,y,z`，可选 `red,green,blue`）
- 可选 YAML 参数文件（`--config`）

输出：

- 一个初始化 PLY（`--output_path`），字段兼容 `core/gaussian_model.py` 的 `load_ply`

## 3. 总体流程

### 3.1 参数解析与配置覆盖

脚本先读取命令行参数，再可选读取 YAML。YAML 中同名键会覆盖参数值，方便固定实验配置。

关键参数含义：

- `downsample_rate`：按时间每 K 帧取 1 帧
- `downsample_points`：每帧点数上限（随机采样）
- `fps`：时间与速度单位换算使用
- `static_vel_threshold`：静态速度阈值
- `max_vel_threshold`：飞点阈值（异常大速度）
- `color_weight`：颜色引导匹配强度
- `smooth_k`：速度中值平滑邻域大小
- `sh_degree`：输出 SH 阶数

### 3.2 帧筛选与读取

按文件名中的数字提取帧号并排序。`start_frame/end_frame` 是相对于磁盘首帧偏移的逻辑索引。

每帧读取后：

- 若点数超限，则随机下采样到 `downsample_points`
- 若无 RGB 字段，颜色回退为灰色（128）

### 3.3 跨帧对应与速度估计

对第 i 帧点云，在第 i+1 帧中做 KNN 最近邻对应（最后一帧速度设 0）。

#### 方法 A：颜色增强 KNN

匹配特征由 `[xyz, sqrt(color_weight)*rgb]` 组成，不只看几何距离。

目的：减少纯几何最近邻导致的错配，尤其在局部几何重复、纹理分区明显的区域。

#### 方法 B：速度单位修正

先算位移 `displacement = p_next - p_curr`，再除以 `dt = downsample_rate / fps` 得到速度：

- `velocity = displacement / dt`

目的：避免把“帧间位移”当成“每秒速度”，减轻时间渲染时的运动拖影。

#### 方法 C：速度中值平滑

对每个点在当前帧内找 K 近邻，对速度向量做中值滤波。

目的：抑制局部错配导致的尖刺速度，减少“飞线”和拉丝伪影。

### 3.4 动静态筛选与去噪

根据速度模长：

- `speed < static_vel_threshold` 视为静态
- `speed > max_vel_threshold` 视为飞点（噪声）

处理规则：

- 飞点全部剔除
- 静态点按 `static_keep_ratio` 随机保留
- 所有静态点速度强制置 0（即便保留）

目的：

- 防止背景点跨帧累积过多，挤占优化容量
- 清理明显错误匹配产生的异常高速点
- 减少静态点残余微动造成的拖影

### 3.5 时间与持续时间初始化

- 时间中心：`t = (frame_idx - target_start_idx) / fps`（秒）
- 时间尺度：`t_scale` 以对数形式存储，当前固定为 `-2`

目的：

- 让训练端在 `normalized_t: false` 下直接用秒单位
- 用统一初值给时域可见性一个可优化起点

### 3.6 空间尺度初始化

- 优先使用 `simple_knn.distCUDA2` 计算局部邻域距离
- 回退到 `torch.cdist`（慢）
- 结果转为 log scale，写入 `scale_0/1/2`

目的：

- 给高斯椭球一个合理的初始空间范围
- 与训练代码的参数化方式（`exp(log_scale)`）一致

### 3.7 SH 与其他参数写出

#### 颜色相关

- `f_dc_*` 由 RGB 转 SH-DC：`(rgb - 0.5) / C0`
- `f_rest_*` 按 `sh_degree` 维度全部置 0 并写入

目的：

- 修复“`sh_degree=3` 直接渲染颜色异常”问题：高阶系数维度完整且初值受控

#### 其他参数

- 旋转：单位四元数 `[1,0,0,0]`
- 不透明度：`logit(0.1)`
- `motion_*`：写入速度向量
- `nx,ny,nz`：当前复用为速度（便于可视化）

## 4. 关键“现象 -> 对策”对应关系

- 现象：初始化后颜色失真，尤其 `sh_degree=3`。
  对策：显式写出完整 `f_rest_*` 并置零，避免高阶 SH 缺失/异常。

- 现象：动态物体出现明显拖影。
  对策：速度按秒归一（除 `dt`），并做局部中值平滑与静态速度置零。

- 现象：动态主体不清晰、背景占比过高。
  对策：静态点按比例压缩，飞点剔除，减少优化早期被背景主导。

## 5. 当前实现的局限与调参建议

局限：

- 对应关系仍是相邻帧单向 KNN，无遮挡显式建模
- 静态判定完全基于速度阈值，未结合语义/可见性
- `t_scale` 当前为常数初始化，未按运动类别自适应

建议优先调参：

- 先调 `static_vel_threshold` 与 `max_vel_threshold`
- 再调 `static_keep_ratio`
- 匹配不稳时提高 `color_weight`
- 噪声速度明显时增大 `smooth_k`

## 6. 最小使用示例

```bash
python tools/process_selfcap_pcd.py \
  --source_path /path/to/dataset \
  --output_path /path/to/init_freetime.ply \
  --fps 60 \
  --downsample_rate 5 \
  --static_vel_threshold 0.08 \
  --max_vel_threshold 2.0 \
  --static_keep_ratio 0.05 \
  --color_weight 0.5 \
  --smooth_k 5 \
  --sh_degree 3
```

也可写入 YAML 后通过 `--config` 加载，保证实验可复现。

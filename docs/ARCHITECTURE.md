# 项目架构文档 (Architecture Documentation)

本文档详细介绍了 Modern Gaussian Splatting 项目的模块化设计与实现细节。该架构旨在解决原始研究代码耦合度高、难以扩展的问题，将其转化为一个结构清晰、易于维护的工程级项目。

## 总体架构设计

项目采用 **分层架构模式 (Layered Architecture)**，将系统划分为以下四个核心层次：

1.  **接口层 (Interface Layer)**: `train.py` 和 `config/`，负责与用户交互，处理输入参数和配置。
2.  **构建层 (Construction Layer)**: `core/builder.py`，负责系统的装配，充当工厂角色。
3.  **核心业务层 (Core Layer)**: `core/`，包含模型、渲染、训练循环、优化和密度控制等核心算法。
4.  **数据基础设施层 (Infrastructure/Data Layer)**: `data/` 和 `utils/`，提供数据加载、图像处理等底层支持。

---

## 模块详细说明

### 1. 配置管理模块 (`config/`)

该模块负责所有实验参数的管理，采用了“配置即代码”与“YAML动态加载”相结合的策略。

*   **`config.py`**: 定义了类型安全的 DataClasses (`DataConfig`, `ModelConfig`, `OptimConfig` 等)。这消除了代码中满天飞的魔法字符串和字典索引，提供了 IDE 自动补全和类型检查支持。
*   **`parser.py`**:
    *   **YAML 解析**: 使用 PyYAML 读取配置文件。
    *   **CLI 合并**: 实现了 CLI 参数对 YAML 配置的覆盖逻辑（优先级：CLI > YAML > Default）。
    *   **配置工厂**: `get_combined_configs()` 函数是获取最终配置的唯一入口。
*   **`default_config.yaml`**: 全局默认配置源，作为所有实验的基准。

### 2. 系统构建模块 (`core/builder.py`)

这是一个典型的 **工厂模式 (Factory Pattern)** 实现，旨在净化 `train.py`。

*   **功能**:
    *   `setup_dataset()`: 根据配置实例化训练集和测试集，处理并行加载 (`num_workers`) 和缓存设备选择 (`cpu` vs `cuda`)。
    *   `setup_model()`: **智能初始化逻辑**的所在地。它不仅创建空模型，还负责自动搜索 `points3D.ply`（支持 COLMAP 输出结构）或执行随机初始化策略。
    *   `setup_optimizer()` & `setup_renderer()`: 标准化的组件装配。

### 3. 核心算法模块 (`core/`)

这是项目的灵魂所在，包含了 3D Gaussian Splatting 的所有算法实现。

*   **`gaussian_model.py` (Model)**:
    *   封装了高斯球的所有可学习参数 (`xyz`, `scaling`, `rotation`, `opacity`, `features_sh`)。
    *   **多模式支持**: 通过继承机制或配置开关，同时支持 **Static (静态)** 和 **FreeTime (时序/4D)** 模式。
    *   实现了参数的激活函数映射（如 `scaling` 用 exp, `rotation` 用 normalize）。
*   **`renderer.py` (Renderer)**:
    *   对 `diff_gaussian_rasterization` 的高层封装。
    *   统一了渲染接口：`render(model, camera, timestamp)`。能够根据输入自动处理静态或动态渲染路径。
*   **`trainer.py` (Trainer)**:
    *   **流程控制器**: 管理整个训练循环（Training Loop）。
    *   集成 TensorBoard 日志记录、进度条显示 (tqdm)、定期 Checkpoint 保存和测试集评估。
    *   它不直接通过 `DataLoader` 强耦合，而是通过 `Sampler` 接口获取数据，增加了灵活性。
*   **`densify.py` (Densification)**:
    *   实现了 **Adaptive Density Control (自适应密度控制)** 算法。
    *   **Clone**: 在梯度大且方差小的区域复制高斯。
    *   **Split**: 在梯度大且方差大的区域分裂高斯。
    *   **Prune**: 剪枝透明度过低或体积过大的无效高斯。
*   **`loss.py`**:
    *   包含 L1 Loss 和 SSIM (Structural Similarity Index) Loss 的组合实现。

### 4. 数据处理模块 (`data/`)

该模块负责将原始的磁盘数据转换为训练所需的内存对象。

*   **`dataset.py` (GaussianDataset)**:
    *   **通用性**: 同时支持 COLMAP (标准 SFM 输出) 和 Blender (NeRF Synthetic) 格式。
    *   **性能优化**: 实现了懒加载 (Lazy Loading) 和 显存缓存 (VRAM Caching) 策略。对于小数据集，可以全量加载到 GPU 显存中以消除 IO 瓶颈。
*   **`camera.py`**:
    *   定义了相机的内参 (Focal, Center) 和外参 (R, T)。
    *   管理投影矩阵的计算与更新。
*   **`samplers.py`**:
    *   **策略模式 (Strategy Pattern)**: 定义了 `DataSampler` 接口。
    *   `StaticSampler`: 用于静态场景，随机抽取图片。
    *   `TemporalSampler`: 用于时序场景，需要同时采样图片及其对应的 `timestamp`。

### 5. 工具模块 (`utils/`)

*   **`general_utils.py`**: 包含数学工具（如四元数转换、Sigmoid 反函数）和通用辅助函数。
*   **`graphics_utils.py`**: 简单的点云结构定义和图形学辅助计算。

---

## 关键流程解析

### A. 初始化流程 (Initialization Flow)
1.  用户运行 `python train.py`。
2.  `config/parser.py` 读取 YAML 并合并 CLI 参数。
3.  `core/builder.py` 扫描数据目录，寻找 COLMAP 的稀疏点云 (`points3D.ply`)。
    *   如果找到 -> 使用 SFM 点云初始化高斯球位置。
    *   如果未找到 -> 在场景包围盒内随机撒点初始化。
4.  创建 `GaussianModel` 并加载到 GPU。

### B. 训练循环 (Training Loop)
1.  **Sample**: 从 `DataSampler` 获取一对 (Camera, Image, Timestamp)。
2.  **Render**: `GaussianRenderer` 将高斯球投影到 2D 图像。
3.  **Loss**: 计算渲染图与 GT 图的 L1 + SSIM Loss。
4.  **Backward**: 反向传播梯度。
5.  **Densify**: (每 N 步) 根据梯度信息执行 Clone/Split/Prune 操作，改变高斯球的数量。
6.  **Optimize**: `GaussianOptimizer` 更新高斯球的属性 (位置、颜色、形状等)。

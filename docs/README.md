# Modern Gaussian Splatting

## 项目简介 (Introduction)

本项目是 **3D Gaussian Splatting** 的现代化重构版本。旨在提供一个更加模块化、易于扩展且结构清晰的代码库，用于训练高质量的静态场景和时序（4D）场景。

核心目标是将原始的"研究型代码"转化为结构良好的"工程级项目"，分离了配置管理、系统构建、模型定义和训练循环，大幅降低了二次开发的难度。

## 核心特性 (Key Features)

*   **架构清晰 (Clean Architecture)**: 彻底解耦了配置解析、系统构建和训练流程。
    *   `train.py` 仅作为轻量级流程控制器。
    *   `config/` 模块专门处理 YAML 解析与参数合并。
    *   `core/` 模块封装了核心组件（Model, Renderer, Trainer）。
*   **多模式支持 (Multi-mode Support)**:
    *   **Static**: 原始 3D Gaussian Splatting，适用于静态场景重建。
    *   **FreeTime (Temporal)**: 支持 FreeTimeGS 模式，用于处理动态/时序场景。
*   **智能初始化 (Smart Initialization)**:
    *   自动检测 COLMAP (`points3D.ply`) 或自定义点云。
    *   若无初始化数据，提供鲁棒的随机初始化策略。
*   **配置管理 (Configuration Management)**:
    *   支持三层配置覆盖：`Default YAML` -> `Custom YAML` -> `CLI Arguments`。
    *   所有配置项通过 `Dataclass` 类型化，减少魔法字符串。
*   **完善的工程设施 (Engineering Utilities)**:
    *   集成 TensorBoard 日志记录。
    *   自动保存/恢复 Checkpoints。
    *   支持 Train/Test 数据集划分与评估。

## 目录结构 (Directory Structure)

项目采用了功能分层的目录结构：

```text
modern_gaussian_splatting/
├── train.py                # 主要入口脚本 (Entry Point)
├── config/                 # 配置管理模块
│   ├── parser.py           # 参数解析与 YAML 读取器
│   ├── config.py           # 配置项 Dataclass 定义
│   └── default_config.yaml # 默认全局配置
├── core/                   # 核心算法实现
│   ├── builder.py          # 系统构建器 (Factory Pattern)
│   ├── trainer.py          # 训练循环与验证逻辑
│   ├── gaussian_model.py   # 高斯模型定义 (Static & FreeTime)
│   ├── renderer.py         # 渲染器封装
│   ├── optimizer.py        # 优化器封装
│   ├── loss.py             # 损失函数 (L1, SSIM)
│   └── densify.py          # 点云克隆/分裂与剪枝逻辑
├── data/                   # 数据加载模块
│   ├── dataset.py          # 通用数据集加载器
│   ├── samplers.py         # 数据采样策略 (Static vs Temporal)
│   └── colmap_loader.py    # COLMAP 数据解析
├── output/                 # 训练输出目录 (Logs, Models)
└── utils/                  # 通用工具函数库
```

## 快速开始 (Quick Start)

### 1. 环境准备

请确保您已安装 PyTorch 和对应的 CUDA 环境。

```bash
# 示例
pip install torch torchvision
pip install -r requirements.txt 
```

### 2. 开始训练 (Training)

**基础用法 (Static Scene):**

最简单的启动方式，只需指定源数据路径：

```bash
python train.py --source_path /path/to/your/dataset
```

**指定输出路径:**

```bash
python train.py --source_path /path/to/dataset --model_path ./output/my_experiment
```

**使用配置文件:**

您可以创建一个自定义 YAML 配置文件来覆盖默认设置：

```bash
python train.py --config config/my_config.yaml
```

### 3. 高级用法

**训练时序模型 (FreeTime 模式):**

```bash
python train.py --source_path /path/to/video_dataset --model_type freetime
```

**覆盖特定参数:**

命令行参数优先级最高，可以覆盖配置文件中的设置：

```bash
# 覆盖迭代次数和 SH 阶数
python train.py --source_path ... --iterations 7000 --sh_degree 0
```

**从断点恢复:**

```bash
python train.py --source_path ... --resume_from output/my_exp/checkpoints/chkpnt30000.pth
```

## 配置系统详解 (Configuration System)

本项目推荐使用 YAML 文件管理实验配置。

`config/default_config.yaml` 包含了所有可配置项的默认值。您可以复制该文件并修改特定部分：

```yaml
# my_config.yaml 示例
data:
  source_path: "/data/nerf_synthetic/lego"
  white_background: true

optim:
  iterations: 30000
  densify_interval: 100

model:
  sh_degree: 3
```


# 3DGS项目改进与解耦分析

> **一句话总结**: 本项目将 3DGS 从一个“学术脚本”进化为了一个“软件工程项目”。

---

## 1. 核心痛点与改进对比

| 维度 | 原始 3DGS 实现 | 本项目 |
| :--- | :--- | :--- |
| **入口脚本** | `train.py` 包含 200+ 行参数解析和复杂的初始化逻辑。 | `train.py` 仅 100 行左右，只负责高层流程控制，逻辑清爽。 |
| **配置管理** | 依赖 `argparse`，参数散落在各个文件中 (`arguments/__init__.py`)，难以追踪和复用。 | 使用 **Dataclass + YAML** (`config/`)。参数强类型化，支持配置文件版本控制。 |
| **系统构建** | 模型、优化器初始化逻辑硬编码在 `train.py` 中，无法复用。 | 引入 **Builder 模式** (`core/builder.py`)，将“构建”与“使用”分离。 |
| **数据加载** | 数据加载逻辑与训练循环耦合。 | 独立的 Dataset 和 Sampler (`data/`)，支持多态采样（静态/时序）。 |
| **扩展性** | 修改模型结构或增加新 Loss 需要侵入式修改 `gaussian_renderer/__init__.py` 或 `scene/__init__.py`。 | 核心组件（Model, Renderer, Trainer）高度模块化，通过继承即可扩展。 |

---

## 2. 详细解耦与重构点

### A. 配置系统的彻底重构 (Configuration Overhaul)

*   **原版问题**: 参数定义分散在 `SceneParams`, `ModelParams`, `PipelineParams` 等多个 helper 类中，且严重依赖命令行参数传递。想查看所有超参数需要跨越多个文件。
*   **本项目改进**:
    *   **统一配置源**: 所有可配置项集中在 `config/config.py` 的 DataClasses 中。
    *   **YAML 驱动**: 支持 `default_config.yaml`，使得实验可复现。以前复现一个实验需要 copy 一长串 CLI 命令，现在只需分享一个 yaml 文件。
    *   **类型安全**: IDE 可以智能提示配置字段，杜绝了 `args.sh_degree` 拼写成 `args.sh_degre` 这类低级错误。

### B. 系统构建与训练逻辑分离 (Builder Pattern)

*   **原版问题**: `train.py` 中充斥着 `scene = Scene(args, gaussian)` 这样的代码。`Scene` 类不仅负责加载数据，还负责初始化点云、处理相机的投影矩阵，甚至还包含了一些保存逻辑。职责过重。
*   **本项目改进**:
    *   **Builder**: 我们创建了 `core/builder.py`。
        *   `setup_dataset`: 专门处理数据加载。
        *   `setup_model`: 专门处理点云初始化（COLMAP/Random）。
    *   **优势**: 如果你想做 Inference 或 Viewer，可以直接调用 `builder.setup_model()` 获取模型，而不需要引入整个 Training Pipeline 的依赖。

### C. 训练器封装 (Trainer Encapsulation)

*   **原版问题**: 训练循环是一个巨大的 `while` 循环，写在 Global Scope 中。变量作用域混乱，TensorBoard 写入、Checkpoint 保存、进度条更新代码交织在一起。
*   **本项目改进**:
    *   **Trainer Class**: 引入 `core/trainer.py::Trainer` 类。
    *   **状态管理**: 将 `current_iteration`, `loss_history` 等状态封装在实例属性中。
    *   **生命周期钩子**: 清晰定义了 `train_step`, `save_checkpoint`, `update_learning_rate` 等方法。
    *   **多模式支持**: Trainer 能够根据 `model.mode` 自动切换 `StaticSampler` 或 `TemporalSampler`，这是原版架构很难做到的（原版通常需要 fork 出一个单独的 repo 来做 4D）。

### D. 数据接口标准化

*   **原版问题**: `Scene` 类直接读取 COLMAP 数据。如果要支持 Blender 数据集或自定义数据格式，需要修改核心库。
*   **本项目改进**:
    *   Dataset 和 Loader 职责分离。
    *   `data/dataset.py` 提供了统一的接口，无论底层是 COLMAP 还是 Blender，吐出来的都是统一的 `Camera` 对象。
    *   这种设计使得接入新的数据集格式（如 Waymo, ScanNet）变得非常简单，只需实现对应的 Loader 逻辑，无需改动渲染核心。

## 3. 优势总结

1.  **更易于二次开发**: 研究人员可以专注于修改 `core/loss.py` 或 `core/gaussian_model.py`，而不用担心破坏数据加载或参数解析逻辑。
2.  **更利于工程落地**: 清晰的模块划分便于将模型导出、部署或集成到现有的 Python 后端服务中。
3.  **调试友好**: 配置、数据、模型构建的每一步都有明确的函数边界，排查 Bug 时可以快速定位是“配置没读对”、“模型没初始化好”还是“训练梯度不对”。


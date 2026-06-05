# 3DGS 训练能耗监测方案

## 环境说明

- CPU：AMD Ryzen 9 9900X（Zen 5，RAPL 通过 powercap 接口暴露）
- GPU：NVIDIA RTX 5070 Ti（NVML 接口）
- OS：RHEL10，SELinux enforcing
- Python 环境：conda `modern_gs`

---

## 一次性权限配置

sysfs 虚拟文件系统不支持 ACL（`setfacl` 会报 `Operation not supported`），需用 `chmod`：

```bash
sudo chmod a+r \
  /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj \
  /sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj
```

验证可读性：

```bash
cat /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
```

重启后失效。持久化方案（SELinux enforcing 下推荐 systemd service，比 udev RUN+= 更可靠）：

```bash
sudo tee /etc/systemd/system/rapl-readable.service << 'EOF'
[Unit]
Description=Make RAPL energy counters world-readable
DefaultDependencies=no
After=local-fs.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'chmod a+r /sys/class/powercap/intel-rapl/*/energy_uj /sys/class/powercap/intel-rapl/*/intel-rapl:*/energy_uj 2>/dev/null || true'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now rapl-readable.service
```

pynvml 安装（无需额外权限）：

```bash
conda run -n modern_gs pip install pynvml
```

---

## EnergyTracker 实现

代码位于项目根目录 `energy_tracker.py`。

```python
"""energy_tracker.py — 训练循环能耗追踪"""
import csv
import os
import time

_RAPL_PKG  = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
_RAPL_CORE = "/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj"
_RAPL_MAX  = 65532610987  # µJ ≈ 65532 J


def _read_uj(path: str) -> int:
    with open(path) as f:
        return int(f.read().strip())


def _delta_uj(after: int, before: int, max_range: int = _RAPL_MAX) -> float:
    """处理计数器溢出，返回 Joules。"""
    raw = after - before if after >= before else max_range - before + after
    return raw / 1e6


class EnergyTracker:
    """
    在训练循环中追踪每个 iteration 的 CPU/GPU 能耗与实时功耗。

    CSV 列：
        iteration, wall_time_s,
        cpu_pkg_energy_J, cpu_core_energy_J, gpu_energy_J,
        cpu_pkg_power_W, gpu_power_W, total_power_W,
        cumulative_cpu_J, cumulative_gpu_J, cumulative_total_J,
        unix_end_us

    用法：
        tracker = EnergyTracker("energy_log.csv")
        tracker.start()
        for i in range(iterations):
            tracker.mark_iter_start(i)
            # ... 训练代码 ...
            tracker.mark_iter_end(i)
        tracker.summary()
        tracker.close()
    """

    def __init__(self, output_path: str, gpu_index: int = 0):
        self.output_path = output_path
        self.gpu_index = gpu_index
        self._handle = None
        self._file = None
        self._writer = None
        self._gpu_energy_supported = True
        self._rapl_available = True
        self._iter_start: dict = {}
        self.last_metrics: dict = {}

        # 累计能耗（Joules）
        self._cum_cpu_j = 0.0
        self._cum_gpu_j = 0.0

    def start(self):
        """初始化 NVML，打开 CSV 文件。"""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            # 探测 energy counter 是否支持
            try:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
            except Exception:
                self._gpu_energy_supported = False
                print("[EnergyTracker] GPU energy counter 不支持，降级为功率采样积分。")
        except Exception as e:
            print(f"[EnergyTracker] NVML 初始化失败: {e}，GPU 能耗不会被记录。")
            self._handle = None

        # 检查 RAPL 可读性
        for path in (_RAPL_PKG, _RAPL_CORE):
            if not os.access(path, os.R_OK):
                self._rapl_available = False
                print(
                    f"[EnergyTracker] 无法读取 {path}，CPU 能耗不会被记录。\n"
                    f"  请先执行：sudo chmod a+r {path}"
                )
                break

        self._file = open(self.output_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "iteration",
            "wall_time_s",
            "cpu_pkg_energy_J",
            "cpu_core_energy_J",
            "gpu_energy_J",
            "cpu_pkg_power_W",
            "gpu_power_W",
            "total_power_W",
            "cumulative_cpu_J",
            "cumulative_gpu_J",
            "cumulative_total_J",
            "unix_end_us",
        ])
        self._file.flush()

    def mark_iter_start(self, iteration: int):
        """在每个 iteration 开始前调用。"""
        entry = {"t": time.perf_counter()}
        if self._rapl_available:
            entry["pkg"] = _read_uj(_RAPL_PKG)
            entry["core"] = _read_uj(_RAPL_CORE)
        if self._handle is not None:
            import pynvml
            if self._gpu_energy_supported:
                entry["gpu_mj"] = pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
            else:
                entry["gpu_power_mw"] = pynvml.nvmlDeviceGetPowerUsage(self._handle)
        self._iter_start[iteration] = entry

    def mark_iter_end(self, iteration: int):
        """在每个 iteration 结束后调用。"""
        t_end = time.perf_counter()
        unix_end = int(time.time() * 1e6)
        s = self._iter_start.pop(iteration)

        wall_time = t_end - s["t"]

        cpu_pkg_j = 0.0
        cpu_core_j = 0.0
        if self._rapl_available:
            cpu_pkg_j = _delta_uj(_read_uj(_RAPL_PKG), s["pkg"])
            cpu_core_j = _delta_uj(_read_uj(_RAPL_CORE), s["core"])

        gpu_j = 0.0
        if self._handle is not None:
            import pynvml
            if self._gpu_energy_supported:
                gpu_end_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
                gpu_j = (gpu_end_mj - s["gpu_mj"]) / 1e3
            else:
                # 梯形积分：平均功率 × 耗时
                gpu_power_mw_end = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                avg_power_w = (s["gpu_power_mw"] + gpu_power_mw_end) / 2.0 / 1e3
                gpu_j = avg_power_w * wall_time

        # 实时功耗 W = J / s（避免除以 0）
        safe_t = max(wall_time, 1e-9)
        cpu_pkg_w = cpu_pkg_j  / safe_t
        gpu_w     = gpu_j      / safe_t
        total_w   = (cpu_pkg_j + gpu_j) / safe_t

        # 更新累计能耗
        self._cum_cpu_j += cpu_pkg_j
        self._cum_gpu_j += gpu_j

        self._writer.writerow([
            iteration,
            round(wall_time, 6),
            round(cpu_pkg_j,  6),
            round(cpu_core_j, 6),
            round(gpu_j,      6),
            round(cpu_pkg_w,  3),
            round(gpu_w,      3),
            round(total_w,    3),
            round(self._cum_cpu_j,                   3),
            round(self._cum_gpu_j,                   3),
            round(self._cum_cpu_j + self._cum_gpu_j, 3),
            unix_end,
        ])
        self._file.flush()

        self.last_metrics = {
            # 每次 iteration 能耗（J）
            "energy/cpu_pkg_J":          cpu_pkg_j,
            "energy/cpu_core_J":         cpu_core_j,
            "energy/gpu_J":              gpu_j,
            "energy/total_J":            cpu_pkg_j + gpu_j,
            # 每次 iteration 实时功耗（W）
            "energy/cpu_pkg_power_W":    cpu_pkg_w,
            "energy/gpu_power_W":        gpu_w,
            "energy/total_power_W":      total_w,
            # 训练至今累计能耗（J）
            "energy/cumulative_cpu_J":   self._cum_cpu_j,
            "energy/cumulative_gpu_J":   self._cum_gpu_j,
            "energy/cumulative_total_J": self._cum_cpu_j + self._cum_gpu_j,
        }

    @property
    def total_energy_J(self) -> float:
        """训练至今的总能耗（J）。"""
        return self._cum_cpu_j + self._cum_gpu_j

    def summary(self):
        """打印整个训练周期的能耗汇总。"""
        total = self._cum_cpu_j + self._cum_gpu_j
        print("\n" + "=" * 50)
        print("Energy Consumption Summary")
        print("=" * 50)
        print(f"  CPU (pkg) : {self._cum_cpu_j:>10.2f} J  ({self._cum_cpu_j/3600:.4f} Wh)")
        print(f"  GPU       : {self._cum_gpu_j:>10.2f} J  ({self._cum_gpu_j/3600:.4f} Wh)")
        print(f"  Total     : {total:>10.2f} J  ({total/3600:.4f} Wh)")
        print("=" * 50)

    def close(self):
        """释放资源。"""
        if self._file:
            self._file.close()
            self._file = None
        if self._handle is not None:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._handle = None
```

---

## 启用方式

通过命令行参数 `--power_monitoring` 开启，默认关闭：

```bash
python train.py --config config/3dgs_test.yaml --power_monitoring
```

或在 YAML 配置文件的 `trainer` 节中设置：

```yaml
trainer:
  power_monitoring: true
```

启用后，`energy_log.csv` 自动保存到 `model_path` 目录。训练结束（或中断）时自动打印汇总：

```
==================================================
Energy Consumption Summary
==================================================
  CPU (pkg) :   1823.45 J  (0.5065 Wh)
  GPU       :  18234.12 J  (5.0650 Wh)
  Total     :  20057.57 J  (5.5715 Wh)
==================================================
```

---

## 输出格式

`energy_log.csv` 包含以下列：

| 列名 | 单位 | 说明 |
|---|---|---|
| `iteration` | — | 训练迭代编号 |
| `wall_time_s` | s | 本次 iteration 实际耗时 |
| `cpu_pkg_energy_J` | J | CPU 封装（整颗芯片）本次 iteration 能耗 |
| `cpu_core_energy_J` | J | CPU 核心域本次 iteration 能耗 |
| `gpu_energy_J` | J | GPU 本次 iteration 能耗 |
| `cpu_pkg_power_W` | W | CPU 封装实时功耗（= cpu_pkg_energy_J / wall_time_s） |
| `gpu_power_W` | W | GPU 实时功耗（= gpu_energy_J / wall_time_s） |
| `total_power_W` | W | CPU + GPU 合计实时功耗 |
| `cumulative_cpu_J` | J | 训练至今 CPU 累计能耗 |
| `cumulative_gpu_J` | J | 训练至今 GPU 累计能耗 |
| `cumulative_total_J` | J | 训练至今 CPU + GPU 累计能耗 |
| `unix_end_us` | µs | iteration 结束的 Unix 时间戳 |

示例（前两行）：

```
iteration,wall_time_s,cpu_pkg_energy_J,cpu_core_energy_J,gpu_energy_J,cpu_pkg_power_W,gpu_power_W,total_power_W,cumulative_cpu_J,cumulative_gpu_J,cumulative_total_J,unix_end_us
1,0.182341,1.243,0.891,2.156,6.817,11.824,18.641,1.243,2.156,3.399,1748910234567890
2,0.179823,1.198,0.856,2.089,6.663,11.617,18.280,2.441,4.245,6.686,1748910234748120
```

---

## TensorBoard 指标

启用 `power_monitoring` 后，以下指标自动写入 TensorBoard：

| 指标路径 | 含义 |
|---|---|
| `energy/cpu_pkg_J` | 每 iteration CPU 封装能耗 |
| `energy/cpu_core_J` | 每 iteration CPU 核心域能耗 |
| `energy/gpu_J` | 每 iteration GPU 能耗 |
| `energy/total_J` | 每 iteration CPU + GPU 总能耗 |
| `energy/cpu_pkg_power_W` | CPU 封装实时功耗 |
| `energy/gpu_power_W` | GPU 实时功耗 |
| `energy/total_power_W` | CPU + GPU 合计实时功耗 |
| `energy/cumulative_cpu_J` | 累计 CPU 能耗（单调递增） |
| `energy/cumulative_gpu_J` | 累计 GPU 能耗（单调递增） |
| `energy/cumulative_total_J` | 累计总能耗（单调递增） |

---

## 分析脚本

将以下代码保存为 `plot_energy.py`：

```python
"""plot_energy.py — 可视化训练能耗"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("energy_log.csv")

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# 每 iteration 能耗（J）
axes[0].plot(df["iteration"], df["cpu_pkg_energy_J"], label="CPU package", alpha=0.8)
axes[0].plot(df["iteration"], df["gpu_energy_J"],     label="GPU",         alpha=0.8)
axes[0].set_ylabel("Energy per iter (J)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 累计能耗（J）— 直接使用 CSV 中已计算好的列
axes[1].plot(df["iteration"], df["cumulative_cpu_J"],   label="CPU cumulative")
axes[1].plot(df["iteration"], df["cumulative_gpu_J"],   label="GPU cumulative")
axes[1].plot(df["iteration"], df["cumulative_total_J"], label="Total", linestyle="--")
axes[1].set_ylabel("Cumulative energy (J)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 实时功耗（W）— 直接使用 CSV 中已计算好的列
axes[2].plot(df["iteration"], df["cpu_pkg_power_W"], label="CPU power (W)", alpha=0.8)
axes[2].plot(df["iteration"], df["gpu_power_W"],     label="GPU power (W)", alpha=0.8)
axes[2].plot(df["iteration"], df["total_power_W"],   label="Total power (W)", alpha=0.6, linestyle="--")
axes[2].set_ylabel("Power (W)")
axes[2].set_xlabel("Iteration")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle("3DGS Training Energy & Power")
plt.tight_layout()
plt.savefig("energy_plot.png", dpi=150)
print("Saved energy_plot.png")
```

运行：

```bash
conda run -n modern_gs python plot_energy.py
```

---

## 注意事项

- **RAPL 计数器溢出**：`max_energy_range_uj` ≈ 65532 J，单次训练通常不会溢出，但 `_delta_uj` 已处理溢出情况。
- **GPU energy counter 支持**：`nvmlDeviceGetTotalEnergyConsumption` 在部分旧驱动/型号上不支持，自动降级为 `nvmlDeviceGetPowerUsage`（瞬时功率，单位 mW）× 耗时的梯形积分。
- **权限不足时的行为**：RAPL 不可读时仅打印警告，GPU 和 wall_time 数据照常记录，CPU 列全为 0，训练不中断。
- **采样开销**：每次 `mark_iter_start/end` 约 10–50 µs（2 次 sysfs 读 + 1 次 NVML 调用），对训练速度影响可忽略。
- **中断安全**：每次 iteration 结束后立即 flush，Ctrl+C 中断不会丢失已完成的数据；`finally` 块确保 `close()` 被调用。
- **多 GPU**：如有多卡，`EnergyTracker` 的 `gpu_index` 参数指定卡号，或扩展为列表在 `mark_iter_start/end` 中循环读取。

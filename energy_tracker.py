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
        cpu_pkg_w  = cpu_pkg_j  / safe_t
        gpu_w      = gpu_j      / safe_t
        total_w    = (cpu_pkg_j + gpu_j) / safe_t

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
            round(self._cum_cpu_j,              3),
            round(self._cum_gpu_j,              3),
            round(self._cum_cpu_j + self._cum_gpu_j, 3),
            unix_end,
        ])
        self._file.flush()

        self.last_metrics = {
            # 每次 iteration 能耗（J）
            "energy/cpu_pkg_J":         cpu_pkg_j,
            "energy/cpu_core_J":        cpu_core_j,
            "energy/gpu_J":             gpu_j,
            "energy/total_J":           cpu_pkg_j + gpu_j,
            # 每次 iteration 实时功耗（W）
            "energy/cpu_pkg_power_W":   cpu_pkg_w,
            "energy/gpu_power_W":       gpu_w,
            "energy/total_power_W":     total_w,
            # 训练至今累计能耗（J）
            "energy/cumulative_cpu_J":  self._cum_cpu_j,
            "energy/cumulative_gpu_J":  self._cum_gpu_j,
            "energy/cumulative_total_J": self._cum_cpu_j + self._cum_gpu_j,
        }

    @property
    def total_energy_J(self) -> float:
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

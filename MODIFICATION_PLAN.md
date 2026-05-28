# 双耳声源定位系统 v3.0 → v4.0 修改方案

> 基于 2026-05-28 架构讨论，聚焦"后方声源（含运动声源）定位性能提升"

---

## 一、问题诊断

### 1.1 当前架构缺陷

现系统 `ActiveLocator`（及 `GCCPHAT` 的 active_head 模式）采用**一次性硬判决**流
程：

```
Phase 1 (0-0.1s): ILD 能量比 → 判左/右（硬判决）
Phase 2 (0.1-0.35s): 头右转15 → ITD斜率 → 判前/后（硬判决）
Phase 3 (0.35s+): GCC-PHAT逐帧跟踪（依赖 Phase 2 的正确性）
```

**核心问题**：Phase 2 一旦误判前后，Phase 3 全链路 180 反转，且**无纠错机制**。

### 1.2 薄弱环节

| 环节 | 失败模式 | 触发条件 |
|------|---------|---------|
| ITD 斜率拟合 | 前后误判 | 房间混响、低 SNR、声源靠近正中面 |
| 运动声源叠加 | 前后误判 | 声源自身运动与头转动产生的 ITD 变化混淆 |
| 后方正中面附近 | 斜率不确定 | ITD 0，斜率拟合 R 趋近 0 |
| 同步转动 | LLR 不收敛 | 主从同步导致 ITD 恒为 0 |

### 1.3 未利用的信息源

- **耳廓频谱陷波**：`hrtf_parametric.py` 已在信号生成端编码了前后 8% 陷波频率偏移（`fb_factor=0.92`），但 `localization` 模块完全未提取此特征
- **逐帧 ITD 累积**：当前仅拟合 250ms 窗口内的单一斜率，未利用后续帧持续更新的似然度

---

## 二、总体方案：LLR 统一框架

### 2.1 核心思路

将 Phase 2 的"一次性硬判决"替换为**持续对数似然比（LLR）跟踪**：

```
H_front: 声源在前半球，方位角 = 胃
H_back:  声源在后半球，方位角 = 180 - 胃（右）或 -180 - 胃（左）

LLR(t) = LLR(t-1) + log P(obs_t | H_back) - log P(obs_t | H_front)
```

- LLR > +threshold 锁定 H_back
- LLR < -threshold 锁定 H_front
- |LLR| < threshold 继续累积证据

### 2.2 三条证据线

每帧贡献的 LLR 增量由三项叠加：

```
LLR = LLR_itd + LLR_ild  + LLR_spectral
```

| 证据线 | 物理原理 | 有效条件 | 当前状态 |
|--------|---------|---------|---------|
| LLR_itd | Woodworth 预测 ITD vs 观测 ITD 偏差 | 头转动且声源偏离正中面 | 有（仅用于斜率） |
| LLR_ild | 前后 ILD 谱模式差异（低频 vs 高频） | 宽带信号 | 部分有（仅 Phase 1） |
| LLR_spectral | 耳廓陷波频率偏移（8%） | 宽带信号，高频段 | **无，需新增** |

### 2.3 自适应阈值

```
threshold = base_threshold / max(1e-3, |ITD_pred_back - ITD_pred_front| / 蟽_itd)
```

- 声源偏侧（ITD 大）两个假设的预测差距大 阈值低 快速锁定
- 声源近正中面（ITD 0）预测差距小 阈值高 谨慎判决

阈值由 Woodworth 模型闭式给出，无额外计算成本。

### 2.4 逃逸机制

当 LLR 持续不收敛（方差过低且绝对值低于阈值），系统判定进入盲区，触发主动
扰动：

```
if stagnation(LLR, window=500ms):
    施加转速扰动（200ms，30/s 方波脉冲）
    扰动后重新评估 LLR 变化趋势
```

仅限旋转自由度内完成，无需额外机械结构。

---

## 三、文件级修改清单

### 3.1 新增文件

#### `src/localization/llr_core.py`（新增）

**职责**：LLR 状态机的纯数学核心，无 I/O，无 Python 类，便于单元测试。

包含函数：
- `predict_itd(azimuth_deg, head_yaw_deg, head_radius, sound_speed) -> float`
  Woodworth 模型闭式预测给定角度和头朝向下的 ITD

- `predict_ild(azimuth_deg, freq_band, head_radius) -> float`
  球体衍射模型预测 ILD

- `compute_discrimination_power(azimuth_lateral, head_yaw_deg, head_radius, sigma_itd) -> float`
  双假设 ITD 预测间距（以 sigma 为单位）用于自适应阈值

- `update_llr(llr_prev, itd_observed, ild_observed, spectrum_observed, azimuth_lateral, head_yaw_deg, fs, head_radius, sigma_itd, sigma_ild, sigma_spectral) -> (float, dict)`
  单帧 LLR 增量计算，返回新 LLR 和分量明细

- `check_stagnation(llr_history, window_frames) -> bool`
  检测 LLR 是否停滞

- `compute_adaptive_threshold(azimuth_lateral, head_yaw_deg, head_radius, sigma_itd, base_threshold) -> float`
  自适应阈值

#### `src/localization/spectral_fb.py`（新增）

**职责**：从宽带双耳信号中提取前后频谱特征。

包含函数：
- `detect_notch_frequencies(frame_left, frame_right, fs, band=(4000, 12000)) -> (float, float)`
  返回左/右耳第一陷波频率估计值

- `spectral_fb_likelihood(left_notch, right_notch, fs) -> (float, float)`
  给定陷波频率，分别输出 P(notch|H_front) 和 P(notch|H_back) 的对数

#### `src/localization/llr_locator.py`（新增）

**职责**：LLR 驱动的定位器，实现 `LocalizationAlgorithm` 接口。

```
class LLRLocator(LocalizationAlgorithm):
    def __init__(self, fs, frame_duration_ms=50.0, frame_hop_ms=25.0,
                 head_radius=0.09, freq_range=None, verbose=True):
        ...
    def localize(self, stereo: np.ndarray) -> LocalizationResult:
        ...
```

**核心流程**：

```
Phase 1 (0-100ms, 4 frames):
    ILD 高频段能量比 判左/右
    初始化 azimuth_lateral = 0.5 * 胃_max (粗粒度，不追求精确)
    初始化 LLR(0)，阈值 threshold_0

Phase 2 (100ms+，持续):
    每帧 (25ms)：
    1. GCC-PHAT itd_obs
    2. ILD 宽带 ild_obs
    3. spectral_fb 陷波位置
    4. update_llr() 计算 LLR
    5. 用 azimuth_lateral 更新 angle_estimate（两个假设各自跟踪）
    6. 用 compute_adaptive_threshold() 动态阈值
    7. check_stagnation()  必要时触发扰动信号

    7a. LLR > +threshold 锁定 H_back
    7b. LLR < -threshold 锁定 H_front
    7c. |LLR| < threshold 继续累积 + 输出带置信度的估计

Phase 3（锁定后）:
    GCC-PHAT 跟踪 + LLR 持续验证（若 LLR 反转到反侧，
    解除锁定并重新评估）
```

**关键设计决策**：

- Phase 1 仅给粗粒度 `azimuth_lateral`（取 0.5 * max_lateral），降低初始误差对后
  续收敛的影响。LLR 自行修正。
- 输出 `LocalizationResult` 填满 `confidence` 字段（每帧的 LLR 归一化置信度），
  供 MATLAB GUI 可视化定位可信度。
- `stagnation` 检测不直接控制电机，而是在 `LocalizationResult` 中新增
  `action_flag` 字段，告知上层"建议施加转速扰动"。**仿真层**（simulator.py）根
  据此标志自主触发扰动。

### 3.2 修改文件

#### `src/localization/base.py`

`LocalizationResult` 新增字段：

```python
confidence: Optional[np.ndarray] = None       # (M,) per-frame LLR-based confidence [0,1]
llr_trajectory: Optional[np.ndarray] = None     # (M,) LLR values over time
llr_components: Optional[dict] = None           # {'itd': [...], 'ild': [...], 'spectral': [...]}
fb_determined_frame: int = -1                   # frame where FB decision made
action_suggestion: Optional[str] = None          # None | 'perturb_rotation' | 'panic_turn'
dual_hypothesis_doa: Optional[np.ndarray] = None # (M,2) [front_doa, back_doa] per frame
```

#### `src/localization/__init__.py`

- `create_localizer()` 新增 `"llr"` 方法
- `active_head=True` 时默认使用 `LLRLocator`（替代当前 `ActiveLocator`）

#### `src/localization/xcorr_itd.py`

- `itd_to_azimuth()` 保持不变
- 新增导出 `inverse_woodworth()` 供 `llr_core.py` 调用
- 新增 `_subsample_peak()` 保持不变

#### `src/localization/gcc_phat.py`

- 提取 `_gcc_phat_single_frame()` 为独立函数，供 `LLRLocator` 复用（消除重
  复代码）

#### `src/pipeline/simulator.py`

- `_overlap_add()` 和 `run()` 中 `active_head` 分支重构
- 当前硬编码的 Phase1/Phase2/flash 时间线移除
- 改为：调用 `LLRLocator.localize()` 获取 `action_suggestion`
- 当 `action_suggestion == 'perturb_rotation'` 时，施加 200ms 方波扰动
- 当 `loc_result.fb_determined_frame >= 0` 时，进入追踪模式
- 仿真级"头朝向声源"的闭环保持现有逻辑

#### `matlab/binaural_gui.m`

- 新增 **LLR 时序曲线**面板（替代当前 Phase 概览的简单文本）
- 新增 **置信度热力图**（x=时间，y=两个假设，颜色=LLR）
- 新增 **双假设 DOA 轨迹对比图**（H_front 和 H_back 各自估计角度，真值叠加）

### 3.3 删除 / 归档

| 文件 | 处理 | 原因 |
|------|------|------|
| `src/localization/active_locator.py` | 标记 `@deprecated`，保留 | 作为对比基准保留 |
| `simulator.py` 中 `_build_yaw_p1()` | 删除 | 硬编码时间线不再需要 |
| `simulator.py` 中 flash/settling 逻辑 | 删除 | 被 LLR 扰动机制替代 |

---

## 四、接口兼容性

### 4.1 场景 YAML

现有 `localization` 段无需改动，仅新增可选字段：

```yaml
localization:
  method: llr              # 新增选项
  active_head: true
  frame_duration_ms: 50.0
  frame_hop_ms: 25.0
  freq_range: [300, 3000]
  llr_base_threshold: 3.0  # 新增：LLR 基础阈值
  enable_spectral: true    # 新增：是否启用频谱线索
  stagnation_timeout_ms: 500  # 新增：盲区逃逸超时
```

### 4.2 向后兼容

- `method: gcc_phat` + `active_head: false` 完全不变（被动模式）
- `method: gcc_phat` + `active_head: true` 行为逐帧一致，仅 LLR 替换硬判决
- `method: xcorr_itd` / `srp_phat` 不变

---

## 五、测试策略

### 5.1 单元测试

| 模块 | 测试内容 |
|------|---------|
| `llr_core.py` | `predict_itd` 对称性（胃 vs -胃）、`update_llr` 单调收敛性、理想
无噪声下收敛速率、自适应阈值在 胃=0 vs 胃=45 下的差异 |
| `spectral_fb.py` | ideal_notch 信号（合成陷波，人工偏移 0.92 倍）：确认正确判别
前后 |
| `llr_locator.py` | 静态前方（LLR 快速收敛至负）、静态后方（LLR 快速收敛至
正）、正中面（LLR 慢收敛，触发 stagnation 标志） |

### 5.2 集成测试

| 场景 | 预期结果 |
|------|---------|
| `front_semicircle.yaml` | LLR 全程为负，无 180 翻转 |
| `circle_once.yaml` | 声源经过后方时 LLR 平稳切换正负，前后交接处无跳变 |
| 后方静止声源 + 噪声 SNR=10dB | 1s 内锁定 H_back，RMSE < 15 |
| 后方运动声源（circle_once 后半圈） | LLR 稳定为正，DOA 轨迹连续 |
| 尾随场景（新增 YAML） | LLR 小幅波动但不长期停滞，spectral 线索持续驱动收敛 |

### 5.3 新增测试场景

```
scenes/rear_static.yaml        # 后方静止声源基准
scenes/rear_moving_linear.yaml # 后方直线运动
scenes/tailing_sync.yaml       # 声源同步绕行（滞后 200ms）
```

---

## 六、风险与注意事项

| 风险 | 缓解措施 |
|------|---------|
| 频谱线索在低 SNR 下不可靠 | 权重自适应：SNR 估计值低时自动降低 LLR_spectral 项权重 |
| Woodworth 模型在极端角度（>135）偏差 | 折叠回 0-90，使用标准公式 |
| LLR 初始化偏差导致收敛慢 | Phase 1 的 ILD 强侧给 LLR(0) 一个小偏移（1.0），在不确定时
略微偏向 H_front（前向先验） |
| 陷波检测在纯音信号上失效 | `is_narrowband` 标志（已在 gcc_phat 实现）关断 LLR_spectral
项 |

---

## 七、实施优先级

| 优先级 | 内容 | 理由 |
|--------|------|------|
| P0 | `llr_core.py` + `llr_locator.py` 核心实现 | 最小可行改动 |
| P0 | `base.py` `LocalizationResult` 字段扩展 | 下游依赖 |
| P1 | `gcc_phat.py` 提取 `_gcc_phat_single_frame()` | 消除代码重复 |
| P1 | `simulator.py` active_head 分支重构 | 集成验证 |
| P2 | `spectral_fb.py` 频谱线索 | 提升后方 SNR 鲁棒 |
| P2 | 新增测试场景 | 回归保护 |
| P3 | MATLAB GUI 新增面板 | 可视化验证 |
| P3 | `active_locator.py` deprecated | 清理 |

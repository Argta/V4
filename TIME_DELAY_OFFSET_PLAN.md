# 时延偏轴跟踪 · 设计方案 v3.0

> 日期：2026-05-30  
> 替代方案：固定几何偏轴（±5°）→ 自适应时延偏轴  
> 核心思想：头追声源过去的位置，运动的源天然产生非零偏角

---

## 一、动机

固定几何偏轴（θ_src + 5°）有两个不优雅的地方：

1. **偏移方向判断**：源在右→偏左，源在左→偏右，源正中→交替。需要显式规则。
2. **静止源也在偏**：源不动时头偏 5°，无必要地制造了 ITD 不对称。

而用户提出的时延思路直接解决了这两个问题：**头追 0.5 秒前的位置，运动的源永远在头的前方，偏角自然存在。**

---

## 二、核心逻辑

`
yaw_target = θ_src(t − τ)

其中 τ 不是固定值，而是由声源速度 ω_src 自适应决定：
    τ = max(τ_min, θ_min / max(|ω_src|, ε))
    τ_min = 0.5s（最小窗口，防止过快震荡）
    θ_min = 5°  （最小偏角，保证 ITD 可用）
    ε = 0.01°/s（防止除零）
`

### 2.1 不同速度下的行为

| ω_src | τ | gap = |ω|·τ | 效果 |
|-------|-----|----------|------|
| 0°/s（静止） | τ_min = 0.5s | 0° | 头对准，不偏——因为不需要 |
| 1°/s | 5s | 5° | 刚好跨过死区 |
| 5°/s | 1s | 5° | 正常跟踪 |
| 10°/s | 0.5s | 5° | 自动收缩 |
| 30°/s | 0.5s | 15° | ITD 充足 |
| 60°/s | 0.5s | 30° | 滞后大但 ITD 极大 |

### 2.2 关键性质

- **gap ≥ θ_min 恒成立**（只要源在动）：死区不出现
- **源静止时 gap = 0**：自然行为，头停在对准位置
- **无方向判断**：不需要知道源在左还是右
- **自适应速度**：快源滞后大但 ITD 也大，慢源滞后小但刚好够

---

## 三、与固定偏轴的对比

| | 固定几何偏轴 | 自适应时延偏轴 |
|---|---|---|
| 方向判断 | 需要（θ_src 符号） | 不需要 |
| 静止源 | 偏 5°，无必要 | 不偏，自然对齐 |
| 慢速源（1°/s） | 偏 5°，OK | 时延拉长到 5s，gap=5° |
| 快速源（30°/s） | 偏 5°，偏小 | 时延 0.5s，gap=15° |
| 正中面交替 | 需要 | 不需要 |
| 实现复杂度 | 简单 | 需环形缓冲 + Kalman ω |

---

## 四、实现方案

### 4.1 数据结构

在 OffsetController 中新增环形缓冲：

`python
# 初始化
self._history_len = int(5.0 / dt)  # 5s 缓冲（覆盖最大 τ）
self._theta_history = np.zeros(self._history_len)
self._hist_ptr = 0  # 写指针

# 每帧写入
self._theta_history[self._hist_ptr] = theta_src
self._hist_ptr = (self._hist_ptr + 1) % self._history_len
`

### 4.2 延迟读取

`python
def _read_delayed(self, delay_frames: int) -> float:
    """读取 delay_frames 帧之前的 θ_src。"""
    if delay_frames >= self._history_len:
        delay_frames = self._history_len - 1
    idx = (self._hist_ptr - delay_frames) % self._history_len
    return self._theta_history[idx]
`

### 4.3 核心算法

`python
def get_yaw_target(self, theta_src: float, omega_src: float) -> float:
    # 写入历史
    self._theta_history[self._hist_ptr] = theta_src
    self._hist_ptr = (self._hist_ptr + 1) % self._history_len

    # 计算自适应时延
    abs_omega = max(abs(omega_src), 1e-2)
    tau = max(self.tau_min, self.theta_min / abs_omega)
    delay_frames = int(tau / self._dt)

    # 读取延迟后的 θ_src
    target = self._read_delayed(delay_frames)
    return target
`

### 4.4 与 Kalman 配合

Kalman 在两种模式下都正常更新：

- **偏轴跟踪态**：θ_local 仍然非零（源在动时），Kalman 正常 update
- **静止态**：θ_local → 0，Kalman 仍可 update（ITD=0 但先前协方差和观测残差仍提供信息）
- **轮盘接管态**：同偏轴跟踪态

---

## 五、静止态的特殊处理

当 |ω_src| < 0.5°/s 且连续 1s 以上：

- τ 自动收敛到 τ_min（0.5s）
- gap = 0° → θ_local = 0°
- 头不偏，定位器在死区
- **但声源也没动，所以不需要新信息**

这是正确行为。状态机进入 STEADY 后无需主动微扰——只有 ω_src 重新增大时才退出 STEADY。可以去掉原有的 300ms 交替微扰逻辑。

---

## 六、放弃的旧逻辑

| 删除 | 原因 |
|------|------|
| compute_offset(theta_src) | 被时延取代 |
| θ_src>0→-5°, θ_src<0→+5° 方向规则 | 不需要 |
| bs(θ_src)<1°→交替±5° 正中面规则 | 不需要 |
| STEADY 态 300ms ±2° 微扰 | 静止时不需要偏 |

---

## 七、文件级修改

| 文件 | 改动 |
|------|------|
| src/tracking/offset_controller.py | 重写 get_yaw_target()：环形缓冲 + 自适应 τ |
| src/tracking/offset_controller.py | 删除 compute_offset()、_perturbation_* |
| src/tracking/offset_controller.py | state_transition 中删除微扰逻辑，STEADY 退出条件仅剩 ω_src 增大 |
| 其他文件 | 不动——head_controller.py 仍调 get_yaw_target(theta_src)，接口不变 |

---

## 八、验证场景

| 场景 | 预期 |
|------|------|
| 静止 500Hz @ 90° | 头追上 90°，θ_local→0（不偏） |
| 慢速 1°/s @ 90°→80° | τ≈5s，头滞后~5°，ITD 始终非零 |
| 中速 10°/s | τ≈0.5s，gap=5°，正常跟踪 |
| 快速 60°/s | τ=0.5s，gap=30°，ITD 极大，滞后可接受 |
| left_to_right 跨正中面 | 无需交替逻辑，时延偏角自然穿越 |

---

*本意见书为 v3.0 偏轴方案，核心改进：用「时间」换「空间」，消除方向判断和交替逻辑。*

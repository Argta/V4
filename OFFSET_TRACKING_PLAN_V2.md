# 偏轴跟踪定位 · 完整设计方案

> 日期：2026-05-30  
> 目标：解决「正中面死区」和「对准即停止」两个结构性问题，建立完整的偏轴跟踪 + Kalman 预测 + 头控系统  
> 代码库：C:\Users\LENOVO\Desktop\dingwei

---

## 一、问题定义

双耳定位系统存在两个根深蒂固的结构性问题，直接限制了持续追踪能力。

### 1.1 正中面死区

当声源位于头部正前方（θ_local → 0°）：

- ITD → 0 μs，ILD → 0 dB，d(ITD)/dθ → 0
- 定位器在 ±3° 内无法分辨方向
- 头对准声源后，反而进入了信息量最低的盲区

这不是算法精度问题，而是双耳线索的物理极限——**0° 是定位器最盲的角度**。

### 1.2 对准即停止

当前 simulator.py 第 238、265 行的逻辑：

`
stop_thresh = 15.0           # |doa| < 15° → 停止转头 → 仿真截断
stopped = True; stop_sample = k   # 对准 = 任务完成
`

隐含假设是"对准 = 0° = 成功"。但在双耳系统中，0° 恰恰是最不该追求的目标。代码在第 277 行直接截断源信号——仿真消失，跟踪中止。

### 1.3 当前代码现状

`
src/
├── localization/
│   ├── base.py            # 无私程接口 (无 process_frame / reset / FrameResult)
│   ├── gcc_phat.py         # 仅 localize() 批量，无 process_frame()
│   ├── llr_locator.py      # 仅 localize() 批量，无 process_frame()
│   └── ...
├── pipeline/
│   └── simulator.py        # active_head 用 stop_thresh=15° + stopped 截断
├── tracking/               # 不存在
└── ...
`

---

## 二、方案核心思想

### 2.1 偏轴跟踪

**头不追 0°，追 θ_src + offset。**

| 声源侧 | offset | 原因 |
|--------|--------|------|
| θ_src > 0（右侧） | −5° | 头偏左，制造耳间交叉，ITD 增大 |
| θ_src < 0（左侧） | +5° | 头偏右，制造耳间交叉 |
| θ_src ≈ 0（正中） | ±5° 交替 | 防止死锁 |

效果：头坐标系 θ_local 始终在 ±5° 附近，ITD 始终非零。**死区从结构上被消除。**

### 2.2 Kalman 跟踪器

作用是平抑 GCC-PHAT 逐帧抖动 + 短期预测。

`
状态:  [θ_src, ω_src]          声源方位角 + 角速度
观测:  θ_local（来自定位器）     θ_local = θ_src − yaw + offset
控制:  yaw（已知，电机反馈）     offset（已知，策略输出）
`

yaw 和 offset 都已知确定——Kalman 只需估计 θ_src 和 ω_src，是标准 2 态恒速模型。

### 2.3 流式接口——偏轴的前提

偏轴跟踪要求每帧都能拿到定位结果和实时 yaw。当前代码只有 localize(整段) 批量接口，必须先补齐流式路径：

`
chunk (480,2) ──→ process_frame(chunk, yaw_head) ──→ doa, confidence（即时输出）
`

**基准接口签名**：

`python
class LocalizationAlgorithm:
    def reset(self):
        """清空内部状态，开始新的流式会话。"""

    def process_frame(self, frame: np.ndarray, yaw_head: float = 0.0):
        """处理单帧。返回 (doa_world_deg, confidence)。"""
`

这个接口对头控系统完全透明——只出 doa_world，只需要 yaw_head。

---

## 三、系统架构

`
┌── 定位系统 ───────────────────────────────────────┐
│                                                     │
│  process_frame(frame, yaw_head)                     │
│    → doa_world, confidence, itd_us, ild_db          │
│                                                     │
└──────┬──────────────────────────┬──────────────────┘
       │ doa_world, confidence    │ yaw_head ◄──┐
       ▼                          │             │
┌── 偏轴跟踪控制器 ────────────────┴──────────────┐
│                                                  │
│  ┌──────────────┐   ┌───────────────────┐       │
│  │Kalman 跟踪器  │   │ OffsetController  │       │
│  │              │   │ 状态机 + 偏轴策略  │       │
│  │ θ_src, ω_src│   │ SEARCH/TRACKING/  │       │
│  │              │   │ STEADY/STOP       │       │
│  └──────┬───────┘   └───────┬───────────┘       │
│         │                   │                    │
│         └───────┬───────────┘                    │
│                 ▼                                │
│         yaw_target = θ_src + offset(θ_src)       │
│                 │                                │
│            PID + 转速约束                        │
│                 │                                │
│            yaw_actual ──────────────────────────►│
└──────────────────────────────────────────────────┘
`

---

## 四、状态机

`
              ┌──────────┐
  系统启动 → │  SEARCH   │ 粗定位，头转动搜索
              └─────┬────┘
                    │ |θ_local| < 30°，持续 0.5s
                    ▼
              ┌──────────┐
              │ TRACKING  │ Kalman 在线，头追 θ_src + offset
              └──┬───┬───┘
                 │   │ Kalman 收敛 + |ω_src| < 10°/s
                 │   ▼
                 │  ┌──────────┐
                 │  │  STEADY  │ 声源静止，每 300ms 微扰维持 ITD
                 │  └────┬─────┘
                 │       │ |ω_src| > 20°/s（声源再次移动）
                 │       └──→ TRACKING
                 │
                 └── 信号丢失 > 1s ──→ ┌──────────┐
                                       │   STOP   │ 头停在当前朝向
                                       └────┬─────┘
                                            │ 信号恢复
                                            ▼
                                          SEARCH
`

---

## 五、Kalman 细节

### 5.1 模型

`
状态转移 (CV):   θ_src(t+1) = θ_src(t) + ω_src(t) · dt
                 ω_src(t+1) = ω_src(t)

观测:            z = θ_local + yaw − offset = θ_src + noise
`

### 5.2 自适应机制

| 条件 | 行为 |
|------|------|
| 观测残差 > 20° | 过程噪声 Q × 50，快速跟上声源急转 |
| confidence < 0.3 | 测量噪声 R / confidence，弱观测权重降低 |
| 残差恢复 | Q 以 0.95 衰减回 base |

### 5.3 预测桥接

`
过渡期（轮盘释放 / offset 切换方向 / 声源短暂丢失）:
  → Kalman 用 predict(dt) 纯预测
  → 最大预测时间 300ms
  → 300ms 后若无有效观测，进入 STOP
`

---

## 六、文件级实施计划

### 6.1 新增文件

| 文件 | 职责 | 预估行数 |
|------|------|----------|
| src/tracking/__init__.py | 模块入口 | ~15 |
| src/tracking/kalman_tracker.py | 2 态 CV Kalman 滤波器，自适应 Q/R | ~150 |
| src/tracking/offset_controller.py | 偏轴策略 + 状态机 A/B/C/D | ~200 |
| src/tracking/head_controller.py | 头控中枢：PID + 双模式仲裁 + 转速约束 | ~250 |

### 6.2 修改文件

| 文件 | 改动 | 风险 |
|------|------|------|
| src/localization/base.py | 新增 FrameResult、eset()、process_frame() | 低——纯增量 |
| src/localization/gcc_phat.py | 实现 process_frame()，提取 _gcc_phat_single_frame() | 中——需保证离线不变 |
| src/localization/llr_locator.py | 实现 process_frame() + LLR 状态维护 | 中 |
| src/pipeline/simulator.py | 删除 stop_thresh/stopped 截断；yaw 更新接入 tracking 模块 | 中 |

### 6.3 不动

| 文件 | 原因 |
|------|------|
| src/localization/xcorr_itd.py | 核心算法不改 |
| src/signals/ 全部 | 不感知头控 |
| src/spatial/ 全部 | 不感知头控 |
| src/evaluation/ 全部 | 不感知头控 |
| 	ools/ 全部 | 离线工具不受影响 |

---

## 七、实施路线

`
Phase 1 ─ 基础层（~2 天）
  ├─ base.py: FrameResult + reset() + process_frame() 抽象
  ├─ gcc_phat.py: _gcc_phat_single_frame() 提取 + process_frame() 实现
  └─ 验证: 离线 localize() vs 流式逐帧 process_frame() 结果一致

Phase 2 ─ 跟踪层（~2 天）
  ├─ kalman_tracker.py: 2 态 CV + 自适应 Q/R + predict()
  ├─ offset_controller.py: 偏轴策略 + SEARCH/TRACKING/STEADY/STOP
  └─ 验证: 仿真中 Kalman 跟踪正弦声源，误差 < 5°

Phase 3 ─ 控制层（~1.5 天）
  ├─ head_controller.py: PID + 双模式 + 转速约束
  ├─ simulator.py: 接入 tracking 模块，删除 stop_thresh 截断
  └─ 验证: active_head=True 仿真全程不"对准即停止"

Phase 4 ─ GUI（~2 天）
  ├─ binaural_gui.m: 新增 yaw 轮盘 Slider + 实时数据面板
  ├─ 轮盘接管/释放回调
  └─ 验证: 拖动轮盘 → 头转动 → θ_world 正确刷新

Phase 5 ─ LLR 补齐（~1 天）
  └─ llr_locator.py: process_frame() + LLR 逐帧状态维护
`

---

## 八、需删除的旧代码

| 位置 | 删除 | 原因 |
|------|------|------|
| simulator.py ~L238 | stop_thresh = 15.0 | 被状态 C (STEADY) 替代 |
| simulator.py ~L245 | stopped = False | 被状态 D (STOP) 替代 |
| simulator.py ~L265 | if abs(diff) < stop_thresh: stopped = True | 不再追求 0° |
| simulator.py ~L277 | 源信号截断 source_signal[:stop_sample] | 不再截断 |

---

## 九、边缘情况

| 场景 | 行为 |
|------|------|
| 声源 180° 急转 | Kalman 残差跳变 → Q 自适应放大 → 快速重新收敛 |
| 声源正中面静止 | STEADY 态，±5° 交替微扰，ITD 永不归零 |
| 声源消失 | 1s 超时 → STOP，头停在当前朝向 |
| 信号恢复 | STOP → SEARCH 重新粗定位 |
| 声源加速运动 | 常速模型滞后 → 残差检测 → Q 自适应 |
| 轮盘接管后声源移动 | Kalman 持续 update（θ_local 仍在产出），释放后立即追新位置 |
| 背后声源 | LLR 前后判定 → fb_is_back → DOA 自动镜像到前半球 |

---

## 十、向后兼容性

| 现有路径 | 是否受影响 |
|----------|:--:|
| un.py 离线仿真（非 active_head） | 否 |
| 	ools/localize_wav.py WAV 定位 | 否 |
| 	ools/session_viewer.py 会话回放 | 否 |
| 	ools/phase1_batch.py 批量测试 | 否 |
| MATLAB GUI 加载 .mat 回放 | 否 |
| simulator.py active_head=True | **是——将接入 tracking 模块** |

---

*本意见书是当前阶段的设计起点。Phase 1 到 Phase 3 之间有明确的依赖关系，但 Phase 4（GUI）和 Phase 5（LLR）可以独立开发、分别合并。*

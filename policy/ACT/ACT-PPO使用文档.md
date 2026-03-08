# ACT-PPO: 基于PPO强化学习的ACT策略微调

## 1. 概述

ACT-PPO 是对预训练 ACT（Action Chunking Transformer）策略的强化学习微调方案。基于 ACT-RL 论文的设计思路，使用 PPO（Proximal Policy Optimization）算法，在保留预训练权重的基础上，仅冻结 ResNet18 视觉编码器，Transformer 编码器/解码器、action_head（μ）、log_std_head（σ）、value_head（V）均为可学习参数。同时维护一个完全冻结的 ACT 副本作为参考策略，通过 KL 散度约束微调幅度。

### 核心设计理念

- **Chunk 级别的 RL**：将 ACT 输出的完整动作序列（50步 × 14维）视为一个 RL 宏动作，整体执行后获得奖励
- **双模型架构**：PPO-ACT（可学习）+ ACT-frozen（完全冻结的参考策略），两者之间计算 KL 散度
- **冻结视觉编码器**：仅冻结 ResNet18 backbone，Transformer + action_head + 新增头均可训练
- **稀疏奖励**：成功 +1.0，失败 -1.0，中间步骤奖励为 0
- **KL 约束**：通过 KL(π_PPO || π_frozen) 惩罚项，防止微调后的策略偏离原始 ACT 分布过远

---

## 2. 文件结构

```
policy/ACT/
├── act_ppo_model.py    # ACT-PPO 模型架构（值函数头、方差头、冻结骨干）
├── ppo_algorithm.py    # PPO 算法核心（GAE、PPO损失、KL散度）
├── ppo_rollout.py      # Rollout 数据收集与环境交互工具
├── ppo_config.yml      # PPO 超参数配置文件
├── train_ppo.py        # 训练主脚本
├── train_ppo.sh        # 训练启动脚本
├── eval_ppo.py         # 评估脚本
└── eval_ppo.sh         # 评估启动脚本
```

---

## 3. 启动方法

### 3.1 前置条件

1. 已完成 ACT 行为克隆预训练，权重保存在：
   ```
   policy/ACT/act_ckpt/act-{task_name}/{task_config}-{expert_data_num}/policy_last.ckpt
   ```
2. 对应的数据集统计信息文件存在：
   ```
   policy/ACT/act_ckpt/act-{task_name}/{task_config}-{expert_data_num}/dataset_stats.pkl
   ```
3. RoboTwin 仿真环境可正常运行（SAPIEN 渲染已配置）

### 3.2 训练启动

在 `policy/ACT/` 目录下执行：

```bash
bash train_ppo.sh <task_name> <task_config> <expert_data_num> <seed> <gpu_id>
```

示例（以 `stack_bowls_two` 任务为例）：

```bash
bash train_ppo.sh stack_bowls_two demo_clean 50 0 0
```

参数说明：

| 参数 | 含义 | 示例 |
|------|------|------|
| `task_name` | 任务名称 | `stack_bowls_two` |
| `task_config` | 任务配置名 | `demo_clean` |
| `expert_data_num` | 专家数据数量 | `50` |
| `seed` | 随机种子 | `0` |
| `gpu_id` | GPU 编号 | `0` |

训练产出保存在：
```
policy/ACT/act_ckpt/act_ppo-{task_name}/{task_config}-{expert_data_num}/
├── policy_best.ckpt          # 评估成功率最高的权重
├── policy_last.ckpt          # 最终迭代权重
├── policy_iter_{N}.ckpt      # 每隔 save_freq 迭代保存的检查点
└── training_log.txt          # 训练日志
```

### 3.3 评估启动

在 `policy/ACT/` 目录下执行：

```bash
bash eval_ppo.sh <task_name> <task_config> <ckpt_setting> <expert_data_num> <seed> <gpu_id>
```

示例：

```bash
bash eval_ppo.sh stack_bowls_two demo_clean demo_clean 50 0 0
```

评估脚本会加载 PPO 微调后的权重，在 100 个 episode 上进行确定性（deterministic）推理，输出成功率。

### 3.4 超参数配置

所有 PPO 超参数在 `ppo_config.yml` 中配置，关键参数：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `clip_ratio` | 0.1 | PPO 裁剪系数 ε |
| `gamma` | 0.995 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE 平滑参数 λ |
| `lr` | 3e-5 | 学习率 |
| `kl_beta` | 0.05 | KL 散度惩罚系数 |
| `vf_coef` | 0.5 | 值函数损失系数 |
| `update_epochs` | 4 | 每轮 rollout 的 PPO 更新轮数 |
| `n_minibatches` | 4 | 每轮更新的 minibatch 数 |
| `num_episodes_per_iter` | 4 | 每次迭代收集的 episode 数 |
| `total_iterations` | 500 | 总训练迭代次数 |
| `ref_std` | 0.01 | 参考策略的固定标准差 |
| `success_reward` | 1.0 | 成功奖励 |
| `failure_reward` | -1.0 | 失败奖励 |
| `num_workers` | 1 | 并行收集/评估的最大worker数量（1=串行，>1=多进程并行） |

---

## 4. 计算原理

### 4.1 模型架构

ACT-PPO 采用双模型架构：PPO-ACT（可学习）和 ACT-frozen（参考策略）。

```
┌─────────────────────────────────────────────────────┐
│                PPO-ACT (可学习)                       │
│                                                     │
│  输入: qpos (1, 14), images (1, 3, 3, 480, 640)     │
│                      │                              │
│  ┌───────────────────▼──────────────────┐           │
│  │ ResNet18 视觉编码器 ×3cam [冻结]       │           │
│  └───────────────────┬──────────────────┘           │
│                      │ detach (梯度截断)              │
│  ┌───────────────────▼──────────────────┐           │
│  │ Transformer Encoder [可学习]           │           │
│  └───────────────────┬──────────────────┘           │
│  ┌───────────────────▼──────────────────┐           │
│  │ Transformer Decoder [可学习]           │           │
│  │ → hidden states (hs) (bs, 50, 512)   │           │
│  └──────┬────────────┬──────────────────┘           │
│         │            │                              │
│    ┌────▼────┐  ┌────▼────┐  ┌────────────┐        │
│    │action_  │  │log_std_ │  │ value_head │        │
│    │head     │  │head     │  │ [可学习]    │        │
│    │[可学习]  │  │[可学习]  │  │hs.mean→V  │        │
│    │→μ       │  │→σ       │  │(bs, 1)     │        │
│    │(bs,50,14│  │(bs,50,14│  └────────────┘        │
│    └─────────┘  └─────────┘                         │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│            ACT-frozen (参考策略, 完全冻结)              │
│                                                     │
│  独立的 DETRVAE 副本，加载相同预训练权重                  │
│  所有参数冻结，始终 eval 模式                           │
│  输出: μ_ref (bs, 50, 14) + σ_ref = 0.01 (固定)     │
└─────────────────────────────────────────────────────┘

KL(π_PPO || π_frozen) 约束微调幅度
```

关键设计点：

- **μ（动作均值）**：来自可学习的 `action_head`，PPO 训练过程中会被更新，使策略能真正优化动作输出
- **σ（动作标准差）**：来自可学习的 `log_std_head`，初始化为 `log(0.01)` ≈ -4.6，接近确定性策略
- **V（状态值）**：来自可学习的 `value_head`，输入为 decoder hidden states 在 chunk 维度上的均值池化
- **梯度截断**：ResNet backbone 输出在进入 Transformer 前做 `detach()`，确保梯度不回传到冻结的视觉编码器
- **BatchNorm 处理**：重写 `train()` 方法，确保冻结的 ResNet18 始终处于 `eval()` 模式，避免 BatchNorm 统计量被破坏
- **参考模型**：`ACTPPOReferenceModel` 是独立的完全冻结 DETRVAE 副本，不与 PPO-ACT 共享权重，输出 `μ_ref` + 固定 `σ_ref=0.01`

### 4.2 策略分布与动作采样

策略输出为对角高斯分布（Diagonal Gaussian）：

```
π(a|s) = N(μ, diag(σ²))
```

其中：
- `μ ∈ R^{50×14}`：可学习的动作均值，来自 PPO-ACT 的 Transformer + action_head（50 个时间步，每步 14 维关节动作）
- `σ ∈ R^{50×14}`：可学习的标准差，由 `exp(clamp(log_std, -5, 0))` 计算

动作采样：
```
a = μ + σ · ε,    ε ~ N(0, I)
```

对数概率计算（对 chunk 和 state_dim 维度求和）：

```
log π(a|s) = Σ_t Σ_d [ -½ ((a_td - μ_td)² / σ_td² + log(σ_td²) + log(2π)) ]
```

### 4.3 Chunk 级别的 Rollout 收集

与传统 RL 的单步交互不同，ACT-PPO 以 **chunk（动作块）** 为单位与环境交互：

```
每个 episode 的交互流程：

1. 观测 s_0 = (qpos, images)
2. 采样动作块 a_0 ~ π(·|s_0)，a_0 ∈ R^{50×14}
3. 逐步执行 a_0 的 50 个动作步
4. 观测 s_1，采样 a_1 ~ π(·|s_1)
5. 重复直到任务成功 / 达到步数上限
6. 稀疏奖励分配：
   - 最后一个 chunk: r = +1.0 (成功) 或 -1.0 (失败)
   - 其余 chunk: r = 0.0
```

每次迭代收集 `num_episodes_per_iter`（默认 4）个 episode，所有 chunk 级别的 transition 存入 `RolloutBuffer`。

### 4.4 GAE 优势估计

使用广义优势估计（Generalized Advantage Estimation）计算每个 chunk 的优势值：

```
δ_t = r_t + γ · V(s_{t+1}) · (1 - done_t) - V(s_t)

A_t = δ_t + (γλ) · (1 - done_t) · A_{t+1}
```

其中：
- `γ = 0.995`：折扣因子（接近 1，因为 chunk 级别的步数较少）
- `λ = 0.95`：GAE 平滑参数，平衡偏差与方差
- `done_t`：episode 结束标志，用于在 episode 边界处截断优势传播

目标回报值：`R_t = A_t + V(s_t)`

### 4.5 PPO 损失函数

总损失由四部分组成：

```
L = L_PPO + c₁ · L_VF + β · L_KL - c₂ · L_entropy
```

#### 4.5.1 PPO 裁剪策略损失 (L_PPO)

```
r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) = exp(log_π_new - log_π_old)

L_PPO = E[ max(-r_t · A_t, -clip(r_t, 1-ε, 1+ε) · A_t) ]
```

- `ε = 0.1`：裁剪范围，限制策略更新幅度
- 取 max 确保损失是保守的下界估计

#### 4.5.2 值函数损失 (L_VF)

带裁剪的值函数损失，防止值函数更新过大：

```
V_clip = V_old + clip(V_new - V_old, -ε_v, ε_v)

L_VF = ½ · E[ max((V_new - R_t)², (V_clip - R_t)²) ]
```

- `ε_v = 0.2`：值函数裁剪范围
- `c₁ = 0.5`：值函数损失系数

#### 4.5.3 KL 散度惩罚 (L_KL)

约束当前策略不偏离原始 ACT 策略过远。两个对角高斯分布之间的 KL 散度：

```
KL(π_θ || π_ref) = Σ_t Σ_d [ log(σ_ref/σ_cur) + (σ_cur² + (μ_cur - μ_ref)²) / (2σ_ref²) - ½ ]
```

关键点：
- `μ_cur` 来自 PPO-ACT 的可学习 Transformer + action_head，训练中会逐渐偏离 `μ_ref`
- `μ_ref` 来自 ACT-frozen 的冻结 action_head，始终保持预训练时的输出
- 参考策略的 `σ_ref = 0.01`（固定值，原始 ACT 无学习的方差）
- KL 惩罚同时约束 `μ_cur` 和 `σ_cur` 不偏离参考策略过远
- `β = 0.05`：KL 惩罚系数

#### 4.5.4 熵正则化 (L_entropy)

对角高斯分布的熵：

```
H(π) = Σ_t Σ_d [ ½ · log(2πe · σ_td²) ]
```

- `c₂ = 0.0`（默认关闭），可通过 `entropy_coef` 配置开启
- 熵奖励鼓励探索，但在冻结骨干 + KL 约束下通常不需要

### 4.6 训练流程

每次 PPO 迭代包含以下步骤：

```
for iteration = 1 to 500:
    ┌─────────────────────────────────────────┐
    │ 1. Rollout 收集                          │
    │    - 收集 4 个 episode 的 chunk 级别数据    │
    │    - 每个 episode: 观测→采样→执行→记录      │
    │    - 稀疏奖励分配到最后一个 chunk            │
    └──────────────────┬──────────────────────┘
                       ▼
    ┌─────────────────────────────────────────┐
    │ 2. GAE 计算                              │
    │    - 计算每个 chunk 的优势值 A_t            │
    │    - 计算目标回报值 R_t = A_t + V(s_t)     │
    │    - 优势值标准化 (zero mean, unit var)     │
    └──────────────────┬──────────────────────┘
                       ▼
    ┌─────────────────────────────────────────┐
    │ 3. PPO 更新 (4 epochs × 4 minibatches)   │
    │    - 随机打乱数据，分成 minibatch           │
    │    - 单次前向传播获取 μ, σ, V               │
    │    - 计算 L_PPO + L_VF + L_KL             │
    │    - 反向传播，梯度裁剪 (max_norm=0.5)      │
    │    - 仅冻结 ResNet backbone，其余均更新    │
    └──────────────────┬──────────────────────┘
                       ▼
    ┌─────────────────────────────────────────┐
    │ 4. 评估 & 保存 (每 10/50 次迭代)           │
    │    - 确定性推理 (μ, 不采样)                 │
    │    - 10 个 episode 计算成功率               │
    │    - 保存最优权重 policy_best.ckpt          │
    └─────────────────────────────────────────┘
```

### 4.7 推理流程

评估/部署时使用确定性推理，不进行动作采样：

```
1. 输入观测 s = (qpos, images)
2. 前向传播获取 μ = PPO-ACT(s)（经过微调的 Transformer + action_head）
3. 直接使用 μ 作为动作（不加噪声）
4. 反归一化：a = μ · action_std + action_mean
5. 逐步执行 50 个动作步
6. 获取新观测，重复
```

注意：推理时不使用时序集成（temporal ensemble），每次输出完整的 50 步动作块并全部执行。

---

## 5. 并行化加速

### 5.1 概述

ACT-PPO 的训练瓶颈主要在 Rollout 收集和策略评估阶段，这两个阶段需要与仿真环境交互，耗时较长且完全串行执行。通过 Python `multiprocessing` 实现这两个阶段的并行化，可以显著提高训练效率。

核心思路：每个 worker 进程创建独立的模型副本（推理模式，不更新参数）和环境实例，并行执行多个 episode 的数据收集或评估。

### 5.2 架构设计

```
主进程 (PPO训练循环)
  │
  ├── 1. 并行 Rollout 收集 (parallel_collect_rollouts)
  │   ├── Worker 0: 收集 episodes [0, k)，seed 范围 [s0, s0+k)
  │   ├── Worker 1: 收集 episodes [k, 2k)，seed 范围 [s0+k, s0+2k)
  │   └── Worker N: 收集 episodes [(N-1)k, ...]
  │   → 合并所有 ChunkTransitions 到 RolloutBuffer
  │
  ├── 2. PPO 更新 (串行，主进程，需要梯度计算)
  │
  └── 3. 并行评估 (parallel_evaluate_policy，每 eval_freq 次迭代)
      ├── Worker 0: 评估 episodes [0, m)
      ├── Worker 1: 评估 episodes [m, 2m)
      └── Worker N: 评估 episodes [(N-1)m, ...]
      → 合并成功率统计
```

### 5.3 配置参数

在 `ppo_config.yml` 中配置：

```yaml
# === Parallelization ===
num_workers: 1    # 并行 worker 数量 (1=串行, >1=多进程并行)
```

| 值 | 行为 |
|----|------|
| `1`（默认） | 使用原始串行代码路径，行为完全不变 |
| `2-4` | 推荐范围，使用多进程并行收集和评估 |
| `>4` | 需要充足的 GPU 显存和 CPU 资源 |

### 5.4 工作原理

1. **模型复制**：主进程将当前模型的 `state_dict` 传递给各 worker，每个 worker 创建独立的 `ACTPPOModel` 并加载权重，设置为 `eval()` 模式
2. **环境隔离**：每个 worker 创建独立的 `TASK_ENV` 仿真环境实例，互不干扰
3. **Seed 分配**：episodes 均匀分配给各 worker，每个 worker 使用不重叠的 seed 范围，避免重复环境
4. **结果合并**：所有 worker 的 transitions/评估结果在主进程中合并

### 5.5 GPU 显存考虑

每个 worker 进程会在 GPU 上加载一份模型用于推理（前向传播），因此：

- **显存占用** ≈ `num_workers × 单模型显存`（推理模式下约 200-400MB/模型）
- **建议配置**：
  - 8GB 显存：`num_workers: 2-3`
  - 16GB 显存：`num_workers: 3-4`
  - 24GB+ 显存：`num_workers: 4-6`

### 5.6 向后兼容性

- `num_workers: 1`（默认值）时，使用原始的 `collect_rollouts()` 和 `evaluate_policy()` 串行函数，行为与修改前完全一致
- PPO 参数更新始终在主进程串行执行，不受 `num_workers` 影响


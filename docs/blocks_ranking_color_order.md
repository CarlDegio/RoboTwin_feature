# 按语言排列 RGB 方块

任务名：`blocks_ranking_color_order`。目标为从左向右排列三个方块，支持
`RGB / RBG / GRB / GBR / BRG / BGR`。这里只要求最终空间顺序，不要求机器人按某个时间顺序抓取。
“左到右”采用场景世界坐标的 x 递增方向，与原 RGB 任务相同。

## 任务与数据约定

- 每个 episode 的 `target_order` 是唯一目标来源，同时决定指令、专家目标位置与成功判定。
  设为 `random` 时，用独立随机数生成器根据场景 seed 选取排列，不消耗场景采样的随机数。
- 同一 seed 可以指定六种不同目标，初始方块位置、大小和目标槽位置保持一致。
  初始场景会排除已经满足任意一种完成条件的布局。
- 专家依次填充左、中、右三个目标槽；实际抓取哪个颜色由目标排列决定。
  成功判定不强制策略复现专家的抓取顺序或精确槽坐标。
- 成功要求颜色顺序正确、相邻 x 间距在 `方块边长 + 3 mm` 到 `13 cm` 之间、
  整行 y 跨度不超过 `3 cm`、方块中心高度距桌面加半边长不超过 `12 mm`，且双夹爪张开。
- `scene_info.json`、规划轨迹和 HDF5 的 `additional_info/task_goal` 保存目标。
  HDF5 该字段为 JSON 字符串，含 `order_id`、`target_order`、`scene_seed`、`reference_frame`。
  `instructions` 保存匹配该目标的 seen 文本；旁边的 instruction JSON 同时保存 seen/unseen 文本。
- 方块由程序创建，无需新增方块素材；机器人、桌面等仍使用已有 RoboTwin assets。

## 采集

以下命令均从仓库根目录、已激活的 `RoboTwin` conda 环境执行。GPU 编号可自行调整。

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/collect_data.py \
  blocks_ranking_color_order color_order --episode-num 600
```

`color_order.yml` 默认按成功 episode 编号循环六种排列，600 条即每种 100 条。
某个 seed 的规划失败时仍重试当前排列，不会因为专家难度不同而打破数量平衡。
建议总条数使用 6 的倍数。默认保存至：

```text
data/color_order/blocks_ranking_color_order/aloha_agilex/
```

小规模端到端检查：

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/collect_data.py \
  blocks_ranking_color_order color_order_smoke
```

该配置只录制 6 条，每种 1 条，且 `save_freq: 60`，采样比正式训练配置稀疏。
训练配置使用 `save_freq: 15`。

可以用 `--target-order BGR` 固定排列，或把配置里的 `balance_orders` 改为 false，
按 seed 随机采样排列。改变目标策略时应复制一个新名字的 task config，使输出目录独立；
续采会校验已有 HDF5/规划轨迹目标，不允许混用不匹配的目标。
`max_seed_attempts` 在采集阶段限制本次运行累计失败的 seed 数，防止无限重试。

## 策略评测

需要先准备兼容该任务、机器人和观测格式的策略 checkpoint。以下路径是占位符：

```bash
bash scripts/eval_policy.sh multitask \
  --config env_cfg/eval/color_order.yml \
  --policy-name Pi_05 \
  --ckpt-name /absolute/path/to/checkpoint \
  --env-cfg-type aloha_agilex \
  --policy-conda-env uv \
  --eval-env-conda-env RoboTwin \
  --task-config color_order \
  --test-num 100 \
  --instruction-type unseen
```

加 `--dry-run` 可先检查调度参数；加 `--target-order BGR` 可仅测试该排列。
`env_cfg/eval/color_order.yml` 默认使用 GPU 7、单 worker，可修改该配置。
若直接使用 `eval_policy_xpolicylab.py`，对应参数拼写为 `--target_order`。

默认随机目标可快速观察混合表现。结果目录除了原有统计，还保存
`color_order_results.json`：每条实际评测 episode 的 seed、排列、指令、成功结果，
以及六种排列各自的样本数、成功率与宏平均。未覆盖的排列成功率为 null；
只有全部六种排列均有样本时才报告宏平均，避免把缺失类别当作零分或忽略它。

做正式对比时，建议六种排列分别指定目标，使用相同 `--seed`、相同测试条数，
然后比较各排列成功率并取等权平均。`--no-expert-check` 可以避免专家筛选造成不同排列
使用不同 seed 集合；环境初始化失败仍可能跳过 seed，应以结果 JSON 中记录的公共 seed
集合做严格配对。先保持单 worker，便于复现 seed 顺序。
评测从 `100000 * (1 + --seed)` 开始取场景 seed，与默认从 0 开始采集的数据分开。

seen/unseen 区分的是指令措辞，不是颜色排列；六种排列都可以训练和测试。
划分训练/验证集时应按 `scene_seed` 分组，同一场景的不同指令或目标不能跨集合。

## 验证与代码入口

```bash
python -m unittest discover -s tests -v

# 另行创建真实 SAPIEN 场景，验证六种目标和评测重置不改变初始布局：
COLOR_ORDER_SIM_TESTS=1 CUDA_VISIBLE_DEVICES=7 \
  python -m unittest discover -s tests -p test_color_order_sim.py -v
```

- `envs/blocks_ranking_color_order.py`：场景、专家与成功判定入口，仅复用原任务的抓放动作。
- `envs/color_order_goal.py`：排列、元数据、成功语义、分排列统计。
- `description/task_instruction/blocks_ranking_color_order.json`：seen/unseen 指令模板。
- `scripts/collect_data.py`：成功样本平衡、回放一致性与续采校验。
- `scripts/eval_policy_xpolicylab.py`：从当前场景取得目标、严格生成指令、写分排列结果。

六条专家采集检查只能证明该任务的规划、回放和数据链路可用，不代表训练后策略的成功率。

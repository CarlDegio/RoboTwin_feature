# pi0.5：五种颜色排布训练与 RGB 留出评测

任务为 `blocks_ranking_color_order`，配置已准备好；训练不要求现在就开始。
两组实验共享同一个 **LeRobot v2.1** 数据集，训练和归一化均排除 RGB。

| 配置名 | 当前输入 | 同任务示范 | 语言指令 |
| --- | --- | --- | --- |
| `pi05_color_order_language` | 三视角图像、当前状态 | 无 | 有 |
| `pi05_color_order_icl` | 三视角图像、当前状态 | 四个时刻，每时刻三视角 | 无 |

共同参数：全量微调，30,000 updates，FSDP=8，global batch=256，horizon=50，
`max_token_len=256`。学习率 warmup 1,000 updates 到 `2e-5`，随后 cosine 衰减，
30,000 updates 到 `2e-6`。保存并保留 `15000/`、`30000/` 两个 checkpoint。
新训练入口默认不会覆盖已有 checkpoint，续训时显式加 `--resume`。

## 数据划分与 ICL 语义

- 训练 / IID：`RBG, GRB, GBR, BRG, BGR`。完整采集每种 100 条时，训练共 500 条。
- OOD：`RGB`，完全不参与梯度训练和 OpenPI 状态 / 动作归一化统计。
- 训练先按排列赋等概率，再在该排列的所有 query 帧中均匀采样；使用有放回加权采样，
  不同轨迹长度或数量不会改变排列的期望权重。每个 batch 不要求精确五等分。
- ICL 每次取 query 时，均匀选择同排列的一条轨迹，允许选自己；按归一化帧位置
  `frame_index / (T-1)`，在 `[10%,30%]`、`[30%,60%]`、`[60%,90%]` 各均匀取一个
  离散帧，再加末帧。各帧使用同步的三个相机。边界帧允许重复；没有合法帧的过短轨迹会报错。
- 示范按时间排列，模型视觉 prefix 顺序固定为当前三视角，然后四个时刻各自的三视角。
  不传示范状态、示范动作、颜色标签或任务语言；当前状态仍遵循 pi0.5 的离散状态 token 编码。
  固定的 `Task: , State: ...; Action:` 格式保留，任务内容为空。
- 评测示范在每个 episode 固定，由评测 seed、颜色排列和实验 seed 决定；不同环境分别缓存。
  只用结构化目标元数据选择示范，不从指令推断排列，元数据不会进入模型。
- RGB 的采集轨迹保留在数据集里，**仅作为 ICL 的 OOD support**，不等于 RGB 训练数据。
  support 排除与当前评测相同 scene_seed 的轨迹。OOD 衡量的是训练未见排列在给定示范后的泛化。
- IID 指新的仿真 episode、已见过的五种排列；不是回放训练轨迹。

`max_token_len` 只包含文本 / 离散状态，不包含图像。224×224 的每张图像产生 256 个
视觉 token。使用官方 tokenizer、六种排列的现有 seen/unseen 模板和 14 维状态逐档检查时，
经典输入最大为 90 tokens，ICL 空任务文本加状态最大为 67 tokens；256 有足够余量。
经典配置有 768 个视觉 token，ICL 有 3,840 个。ICL 的计算与显存需求明显更大，
仍需用实际数据验证 8 卡、batch 256 的训练占用；提高文本上限不能解决视觉 token 的显存问题。

## 环境与转换

在仓库根目录执行。示例沿用默认数据集 ID，路径可按实际采集位置调整。
`lerobot==0.3.3` 对应数据格式 v2.1；0.4.x 写 v3.0。本次固定 0.3.3，并保留此适配器已有的
Torch 2.10 / torchvision 0.25 / torchcodec 0.10 版本；训练使用 JAX，图像数据不走 torchcodec。

```bash
export ROBOTWIN_ROOT="$PWD"
export PYTHONPATH="$ROBOTWIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd XPolicyLab/policy/Pi_05/openpi

# 数据齐备、准备运行时再同步环境；不会自动启动训练。
uv sync --locked --group lerobot --index-url https://repo.huaweicloud.com/repository/pypi/simple

uv run --no-sync scripts/process_color_order.py \
  --raw-dir "$ROBOTWIN_ROOT/data/color_order/blocks_ranking_color_order/aloha_agilex" \
  --repo-id color_order-aloha_agilex-joint-v21
```

转换生成标准 v2.1 图像型 LeRobot 数据及额外 `meta/color_order.json`，记录每条轨迹的
排列、scene_seed、长度、起始帧和来源。图像通过共享 RGB 解码器读取，等比例填充到 224×224。
动作沿用现有 pi05 转换器的下一帧关节状态约定。`--fps=50` 是沿用现有转换器的名义时间戳频率，
不做重采样；训练 horizon 按实际保存帧数计算，评测控制频率使用采集配置的 15 个仿真步。

转换会检查目标与颜色列表一致性、三相机帧数和有效状态；拒绝覆盖已有同名数据集。
训练会要求五个 IID 排列都存在，数据条数无需现在恰好为 500。转换中断或缺少 manifest 的数据集
不会被当作可训练数据。重新转换应选择新 `--repo-id` 或由用户处理不完整输出。

自定义 repo ID 时，需要同时更改两个训练配置的 `data.repo_id` 和 ICL deploy 文件的
`context_repo_id`，然后重新计算各自的归一化统计。

## 归一化与训练

```bash
# 仍在 Pi_05/openpi 中；两组都只使用五种训练排列。
uv run --no-sync scripts/compute_norm_stats.py --config-name pi05_color_order_language
uv run --no-sync scripts/compute_norm_stats.py --config-name pi05_color_order_icl

cd "$ROBOTWIN_ROOT"
bash XPolicyLab/policy/Pi_05/train_color_order.sh language 0 0,1,2,3,4,5,6,7 \
  --weight-loader.params-path=/mnt/base_model/pi05_jax_model/params

# 第一组结束后再启动第二组。
bash XPolicyLab/policy/Pi_05/train_color_order.sh icl 0 0,1,2,3,4,5,6,7 \
  --weight-loader.params-path=/mnt/base_model/pi05_jax_model/params
```

本地权重路径不存在时，省略 `--weight-loader.params-path`，配置默认使用官方 pi05_base 地址。
输出目录分别为 `XPolicyLab/policy/Pi_05/checkpoints/color_order-{language,icl}-aloha_agilex-joint-0/`。
以上是待运行命令，不代表已有训练结果。

## IID / OOD 评测

在具备 RoboTwin 评测依赖的 Python 环境、仓库根目录执行：

```bash
python scripts/eval_pi05_color_order.py language \
  --checkpoint XPolicyLab/policy/Pi_05/checkpoints/color_order-language-aloha_agilex-joint-0/30000 \
  --split all --test-num 100 --seed 0

python scripts/eval_pi05_color_order.py icl \
  --checkpoint XPolicyLab/policy/Pi_05/checkpoints/color_order-icl-aloha_agilex-joint-0/30000 \
  --split all --test-num 100 --seed 0
```

`--test-num` 为每种排列的测试数；`all` 是五个 IID 加一个 OOD，亦可单独选 `iid` 或 `ood`。
`--dry-run` 只验证调度参数。GPU 调度沿用 `env_cfg/eval/color_order.yml`，默认 GPU 7、单 worker。
自动选择对应 deploy 文件，使用相同评测 seed，关闭专家可解性筛选，避免不同排列被专家难度筛选。
默认 `--instruction-type seen`，使经典配置的差别集中在目标排列；也可显式改为 `unseen` 测试措辞。
现有评测器从 `100000 * (1 + seed)` 开始创建新场景。初始化失败仍可能跳过 seed，严格配对比较
以各结果文件共同的 seed 为准。

每个排列保存自己的 `color_order_results.json`。IID 汇总取五种排列成功率的算术平均，
RGB 单独报告为 OOD；不要把六种排列混成一个指标。评测脚本不会自动启动训练。

## 代码验证

轻量测试（只需 NumPy、h5py、pytest）：

```bash
python -m pytest XPolicyLab/policy/Pi_05/tests/test_color_order_dataset.py \
  XPolicyLab/policy/Pi_05/tests/test_color_order_conversion.py tests/test_pi05_color_order_eval.py
```

具备 pi05 环境后，CPU 输入链路检查（首次可能下载 tokenizer）：

```bash
JAX_PLATFORMS=cpu PYTHONPATH="$ROBOTWIN_ROOT:$ROBOTWIN_ROOT/XPolicyLab/policy/Pi_05/openpi/src" \
  XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -m pytest XPolicyLab/policy/Pi_05/tests -q
```

这些检查不替代真实轨迹的数据加载 smoke、完整权重训练、GPU 显存测试与仿真成功率评测。

本次验证：40 项相关测试通过（合成 HDF5→LeRobot v2.1→两种 OpenPI batch、RGB 从统计中排除、
上下文生命周期、视觉前缀、tokenizer、学习率及已有 transform/loader/filter 回归）；六种排列
的评测调度干运行通过。未运行依赖旧 `debug` / `pi0_aloha_sim` 配置的数据加载测试。
主仓库完整 unittest 运行中，`test_color_order_pipeline` 因当前环境缺少 SAPIEN 无法导入，
`test_color_order_sim` 按其 GPU 开关跳过；没有进行真实训练或仿真成功率评测。

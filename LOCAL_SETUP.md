# Local RoboTwin / Pi_05 setup

> 当前 B200 主机（2026-09-28）的路径、激活方式和采集说明见 [docs/setup_b200_20260928.md](docs/setup_b200_20260928.md)。下文保留此前 H20 主机的安装记录。

This checkout is configured for Ubuntu 24.04 and NVIDIA H20. Local changes
are not committed. PyTorch3D is intentionally omitted; RGB collection and
evaluation configurations keep `data_type.pointcloud: false`.

## Environments

- Simulator / collection / evaluation: Conda `RoboTwin`, Python 3.10,
  PyTorch 2.7.1 + torchvision 0.22.1 (CUDA 12.8 wheels), CuRobo v0.7.8.
- Policy: `XPolicyLab/policy/Pi_05/openpi/.venv`, managed by uv using the
  vendored OpenPI lockfile (Python 3.11, JAX 0.5.3, PyTorch 2.10).
- Conda installation: `/mnt/workspace/liuzihao/miniconda3`.
- XPolicyLab is checked out at RoboTwin's pinned commit
  `fa431ecd893ee706883e64fe5fe1464ec8cd928d`.

Activate the simulator in a fresh terminal:

```bash
source /mnt/workspace/liuzihao/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin
cd /mnt/workspace/liuzihao/project/RoboTwin_feature
```

Activate the policy in a separate terminal:

```bash
cd /mnt/workspace/liuzihao/project/RoboTwin_feature
source XPolicyLab/policy/Pi_05/openpi/.venv/bin/activate
```

The Pi_05 evaluation scripts use `conda info --base` even though the model uses
uv. Conda must remain on PATH. PyYAML has been added to base because the policy
launcher uses base Python to read `deploy.yml`.

CUDA 12.8.1 is installed as the `cuda-compiler` and `cuda-libraries-dev`
components from `nvidia/label/cuda-12.8.1`; the optional Nsight profiler is not
installed. NVCC reports 12.8.93 and the environment's C++ compiler is GCC 13.4.
The environment sets `CUDA_HOME` to its Conda prefix and initializes
`NVCC_PREPEND_FLAGS` / `NVCC_APPEND_FLAGS` so NVIDIA's activation script also
works in shells using `set -u`.
It also sets `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`, the system
NVIDIA ICD; SAPIEN otherwise misses this location and substitutes its bundled ICD.

For a fresh installation, explicitly select the CUDA wheels before installing
`scripts/requirements.txt`:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r scripts/requirements.txt
```

The unmodified `scripts/_install.sh` also installs PyTorch3D and updates
XPolicyLab to upstream main. This local setup instead retains the submodule pin
and skips PyTorch3D.

## Smoke commands

Collect one RGB demonstration (validated successfully):

```bash
python scripts/collect_data.py stack_bowls_two smoke_rgb
```

Select a GPU with `CUDA_VISIBLE_DEVICES` before launching the command. Output is
under `data/smoke_rgb/stack_bowls_two/aloha_agilex/`. The completed smoke
episode is `data/episode_0000000.hdf5`; replay video is
`video/episode_0000000.mp4`. Reusing this config skips the existing episode;
use a new config/output location to collect a fresh smoke sample.

For policy evaluation, use a RoboTwin-compatible trained checkpoint and matching
`train_config_name`, normalization asset `repo_id`, and `checkpoint_num` in
`XPolicyLab/policy/Pi_05/deploy.yml`. The smoke scheduler uses GPU 7; edit its
`gpu_ids` to select another GPU.

```bash
bash scripts/eval_policy.sh multitask \
  --config env_cfg/eval/pi05_smoke.yml \
  --policy-name Pi_05 \
  --ckpt-name /absolute/path/to/compatible/checkpoint \
  --env-cfg-type arx_x5 \
  --policy-conda-env uv \
  --eval-env-conda-env RoboTwin \
  --action-type joint \
  --task-config demo_clean \
  --test-num 2 \
  --dry-run
```

Remove `--dry-run` to run the evaluation. A schedule check does not load weights
or measure policy performance.

## Current validation

- NVIDIA Vulkan detection and ray-tracing pipeline extension: passed.
- All three official asset archives downloaded and extracted; embodiment paths
  updated for this checkout.
- Pi_05 adapter/config imports and JAX GPU arithmetic: passed on H20.
- Pi_05 inference path with small random language/action modules: passed,
  producing finite actions of shape `(1, 4, 32)`. This is an environment smoke
  check, not a pretrained checkpoint or benchmark evaluation.
- Pi_05 training data-loader, TorchCodec video-decoder and Rerun imports: passed.
- RoboTwin package installation and dependency consistency: passed (274 packages).
- CuRobo CUDA extensions: built for H20 (`sm_90`); joint-space planning passed
  with a finite 31-frame trajectory.
- SAPIEN ray-traced RGB capture: passed with the system NVIDIA Vulkan ICD.
- End-to-end `stack_bowls_two` collection: passed with seed 1 (seed 0 was
  rejected by the task success check). Saved 307 state/action pairs and a
  308-frame video. Head, both wrist cameras, and the additional front view
  decode as 320x240 RGB; finite values, shifted state/action alignment, and
  nonempty language instructions passed validation.
- Pi_05 evaluation scheduler dry run: passed. No trained policy checkpoint
  was loaded, so policy task success rates are not validated.

The upstream OpenPI lockfile overrides NumPy to 1.26.4, while LeRobot 0.4.4's
`rerun-sdk` dependency requires NumPy >=2. `uv pip check` reports this conflict.
Keep the model's NumPy pin; the tested model inference works. Rerun visualization
is not part of the inference validation.

Persistent validation logs and environment version lists are saved under
`eval_result/setup_20260924/` (ignored by Git). Working installation logs and
verified local package downloads are in `/tmp/robotwin-setup/`.

The bundled Pi_05 `openpi/scripts/process_data.py` still imports the old
`lerobot.common` namespace and does not run with the pinned LeRobot 0.4.4.
For RoboTwin data conversion, use the shared
`XPolicyLab/scripts/transform_lerobot_v30_format.py` workflow documented in the
root README; the model training data-loader already supports the new namespace.

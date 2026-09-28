# Pi05 color-order training implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan inline, task by task.

**Goal:** Prepare language-conditioned and visual-context pi05 experiments for five IID color orders and held-out RGB.

**Architecture:** Convert all six orders to one LeRobot dataset with a validated episode manifest. Filter RGB from training and normalization; weight the five training orders equally. Add four chronological demonstration frames (three cameras each) to the existing image prefix, retaining pi05 discrete current-state input and dropping task text for ICL. Use the same demonstration sampler for inference, fixed per episode.

**Tech stack:** Existing OpenPI JAX/FSDP, LeRobot, NumPy, RoboTwin evaluation scripts.

**Spec:** User request in this session, with the concrete requirements below.

## Global constraints

- IID/train: RBG, GRB, GBR, BRG, BGR; OOD: RGB. Expected complete collection: 100 trajectories/order, 500 training trajectories.
- Full fine-tuning, 30,000 updates, global batch 256, FSDP 8, horizon 50, max text/state tokens 256.
- Warmup 1,000 updates; cosine schedule peak 2e-5 and final 2e-6; checkpoints every 15,000 updates.
- ICL: uniformly choose a same-order trajectory, including self; draw one frame uniformly from each normalized interval [0.1,0.3], [0.3,0.6], [0.6,0.9], plus final frame, chronological order.
- Assumption: all four demonstration frames use all three cameras; evaluation support is fixed per episode and may use collected RGB demonstrations, never evaluation rollouts.
- Decode raw images only through XPolicyLab shared helpers. Preserve existing user changes, including uv.lock. Do not launch training or data collection.

## Review focus

- RGB must be excluded even when all six orders share a dataset and instruction variants.
- Different episode lengths must not bias order weights or demonstration trajectory selection.
- Extra images must survive repacking, normalization, model preprocessing, and inference batching.
- Missing metadata, empty order pools, and short trajectories fail clearly instead of silently leaking labels or changing sampling.
- Evaluation reset and batching must keep support aligned with each episode without passing order/text into the model.

## Tasks

### 1. Dataset and demonstrations

- [x] Write failing tests for manifest validation, RGB filtering, per-order sampling weights, four-frame intervals, and same-order support selection.
- [x] Implement `openpi/training/color_order_dataset.py`, plus strict `scripts/process_color_order.py` conversion and manifest output.
- [x] Integrate filtering/context into `training/data_loader.py`; run sampling tests.

### 2. Model inputs and training presets

- [x] Test that visual input transforms erase language and retain all 15 images with chronological frame keys.
- [x] Add color-order data config, two train configs, optional image-key specification, and ordered prefix preprocessing.
- [x] Add training wrapper using FSDP 8 without existing train.sh override; preserve normal configs.
- [x] Check language/state token budget, schedule and checkpoint semantics; validate with available dependencies.

### 3. Evaluation and documentation

- [x] Add same-order support loading and per-episode caching to Pi_05 adapter; expose deploy config override in server launcher.
- [x] Provide language/ICL deployment configs and IID/OOD evaluation launcher covering each order equally on fresh simulator seeds.
- [x] Add tests for support lifetime, language removal, split dispatch and missing data errors.
- [x] Document conversion, normalization, training and evaluation commands and resource limitations.
- [x] Run focused tests, existing applicable regression suite, syntax checks and final diff review. Report unavailable runtime validation explicitly.

## Validation and decisions

- Work performed in the supplied workspace, preserving the pre-existing mirror-source lockfile edits and untracked setup notes. No training, simulator runs, or commits were started.
- Data format clarified by user as v2.1. Pin LeRobot 0.3.3, preserve the adapter's existing B200 Torch/vision/codec versions through dependency overrides, and regenerate the lock using its existing Huawei mirror.
- Independent read-only review found the v2.1 `add_frame(frame, task=...)` API mismatch. A real synthetic conversion test reproduced it before the converter fix. The same test also caught the existing converter passing an HDF5 Dataset object rather than its materialized bytes to the shared decoder; fixed in that shared conversion helper.
- CPU input tests verify all 15 images, explicit temporal prefix ordering, empty task text with retained state, optimizer schedule, and full-parameter filters. Official tokenizer measurement: existing six-order templates/state at most 90 tokens, ICL state-only format at most 67.
- No real checkpoint training/inference, 8-GPU memory/throughput validation, or simulator success-rate evaluation is claimed. These await the completed dataset and runtime environment.
- Final focused suite: 40 passed, 2 legacy-config tests deselected; six-order evaluation dry-run passed. Root unittest: `test_color_order_pipeline` import blocked by absent SAPIEN; simulator test skipped by its GPU opt-in switch. Python syntax, new-file lint and both repositories' diff whitespace checks passed.

import os
import numpy as np
from dataclasses import dataclass

from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM
from experiments.robot.openvla_utils import (
    get_vla,
    get_processor,
    get_action_head,
    get_proprio_projector,
    get_vla_action,
)


@dataclass
class InferenceConfig:
    pretrained_checkpoint: str
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = True
    use_proprio: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    num_images_in_input: int = 3
    center_crop: bool = True
    unnorm_key: str = ""
    num_open_loop_steps: int = NUM_ACTIONS_CHUNK
    lora_rank: int = 32
    action_mode: str = "absolute"
    delta_gripper: bool = False


def encode_obs(obs: dict) -> dict:
    return {
        "full_image": obs["observation"]["head_camera"]["rgb"],
        "left_wrist_image": obs["observation"]["left_camera"]["rgb"],
        "right_wrist_image": obs["observation"]["right_camera"]["rgb"],
        "state": obs["joint_action"]["vector"],
        "instruction": obs["language"],
    }


class Model:
    def __init__(self, cfg: InferenceConfig):
        if cfg.action_mode not in ("absolute", "delta"):
            raise ValueError(
                f"Unsupported action_mode={cfg.action_mode!r}; expected 'absolute' or 'delta'."
            )
        self.cfg = cfg
        self.vla = get_vla(cfg)
        self.processor = get_processor(cfg)
        self.action_head = None
        if cfg.use_l1_regression or cfg.use_diffusion:
            self.action_head = get_action_head(cfg, self.vla.llm_dim)
        self.proprio_projector = None
        if cfg.use_proprio:
            self.proprio_projector = get_proprio_projector(
                cfg, self.vla.llm_dim, PROPRIO_DIM
            )

    def get_action(self, observation: dict):
        obs = encode_obs(observation)
        actions = get_vla_action(
            cfg=self.cfg,
            vla=self.vla,
            processor=self.processor,
            obs=obs,
            task_label=obs["instruction"],
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            use_film=self.cfg.use_film,
        )
        return actions


def get_model(usr_args: dict):
    config_args = {
        "pretrained_checkpoint": usr_args["checkpoint_path"],
        "use_l1_regression": usr_args.get("use_l1_regression", True),
        "use_diffusion": usr_args.get("use_diffusion", False),
        "use_film": usr_args.get("use_film", True),
        "use_proprio": usr_args.get("use_proprio", True),
        "load_in_8bit": usr_args.get("load_in_8bit", False),
        "load_in_4bit": usr_args.get("load_in_4bit", False),
        "num_images_in_input": usr_args.get("num_images_in_input", 3),
        "center_crop": usr_args.get("center_crop", True),
        "unnorm_key": usr_args["unnorm_key"],
        "num_open_loop_steps": usr_args.get("num_open_loop_steps", NUM_ACTIONS_CHUNK),
        "lora_rank": usr_args.get("lora_rank", 32),
        "action_mode": usr_args.get("action_mode", "absolute"),
        "delta_gripper": usr_args.get("delta_gripper", False),
    }

    cfg = InferenceConfig(**config_args)
    return Model(cfg)


def reset_model(model=None):
    pass


def convert_action_for_env(action: np.ndarray, observation: dict, cfg: InferenceConfig):
    if cfg.action_mode == "absolute":
        return action

    current_action = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    env_action = current_action.copy()
    delta_action = np.asarray(action, dtype=np.float32)

    if cfg.delta_gripper:
        env_action += delta_action
    else:
        env_action[:6] += delta_action[:6]
        env_action[7:13] += delta_action[7:13]
        env_action[[6, 13]] = delta_action[[6, 13]]
        env_action[[6, 13]] = np.clip(env_action[[6, 13]], 0.0, 1.0)

    return env_action


def eval(TASK_ENV, model: Model, observation: dict):
    observation["language"] = TASK_ENV.get_instruction()

    actions = model.get_action(observation)
    for action in actions:
        env_action = convert_action_for_env(action, observation, model.cfg)
        TASK_ENV.take_action(env_action)
        observation = TASK_ENV.get_obs()

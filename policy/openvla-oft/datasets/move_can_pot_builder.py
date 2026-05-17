from typing import Iterator, Tuple, Any
import os
import h5py
import glob
import numpy as np
import tensorflow_datasets as tfds
import random
from datasets.conversion_utils import MultiThreadedDatasetBuilder
tfds.core.constants.DATA_DIR = "/mnt/tensorflow_datasets"

ACTION_MODE = os.environ.get("OPENVLA_ACTION_MODE", "absolute").lower()
DELTA_GRIPPER = os.environ.get("OPENVLA_DELTA_GRIPPER", "false").lower() in (
    "1",
    "true",
    "yes",
)
GRIPPER_INDICES = (6, 13)
if ACTION_MODE not in ("absolute", "delta"):
    raise ValueError(
        f"Unsupported OPENVLA_ACTION_MODE={ACTION_MODE!r}; expected 'absolute' or 'delta'."
    )


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    print(
        f"[INFO] Generating examples from {len(paths)} paths "
        f"(action_mode={ACTION_MODE}, delta_gripper={DELTA_GRIPPER})"
    )
    for path in paths:
        print(f"[INFO] Parsing file: {path}")
        with h5py.File(path, "r") as f:
            required_keys = [
                "/relative_action",
                "/head_camera_image",
                "/left_wrist_image",
                "/right_wrist_image",
                "/low_cam_image",
                "/action",
                "/seen",
            ]
            if not all(k in f for k in required_keys):
                for key in required_keys:
                    if key not in f:
                        print(f"[ERROR] Missing key: {key} in {path}")
                print(f"[WARNING] Missing expected keys in {path}, skipping")
                continue
            T = f["/action"].shape[0]
            absolute_actions = f["/action"][:].astype(np.float32)
            if ACTION_MODE == "absolute":
                actions = absolute_actions[1:].copy()  # (T-1, 14)
            else:
                actions = f["/relative_action"][: T - 1].astype(np.float32)
                if not DELTA_GRIPPER:
                    actions[:, GRIPPER_INDICES] = absolute_actions[1:, GRIPPER_INDICES]
            head = f["/head_camera_image"][ : T-1 ].astype(np.uint8)
            left = f["/left_wrist_image"][ : T-1].astype(np.uint8)
            right = f["/right_wrist_image"][ :T-1].astype(np.uint8)
            low = f["/low_cam_image"][ : T-1].astype(np.uint8)
            states = absolute_actions[: T - 1]  # (T-1, 14)
            seen = [
                s.decode("utf-8") if isinstance(s, bytes) else s for s in f["/seen"][()]
            ]
            T -= 1

            if not seen:
                print(f"[ERROR] No 'seen' instructions found in {path}")
                continue

            if not (
                head.shape[0]
                == left.shape[0]
                == right.shape[0]
                == low.shape[0]
                == T
                == states.shape[0]
            ):
                print(f"[ERROR] Data length mismatch in {path}")
                continue

            instruction = seen

            steps = []
            for i in range(T):
                step = {
                    "observation": {
                        "image": head[i],
                        "left_wrist_image": left[i],
                        "right_wrist_image": right[i],
                        "low_cam_image": low[i],
                        "state": states[i],
                    },
                    "action": actions[i],
                    "discount": np.float32(1.0),
                    "reward": np.float32(1.0 if i == T - 1 else 0.0),
                    "is_first": np.bool_(i == 0),
                    "is_last": np.bool_(i == T - 1),
                    "is_terminal": np.bool_(i == T - 1),
                    "language_instruction": instruction,
                }
                steps.append(step)

            print(f"[INFO] Yielding {len(steps)} steps from {path}")
            yield path, {"steps": steps, "episode_metadata": {"file_path": path}}


class robotwin4stack_rand_aloha(MultiThreadedDatasetBuilder):
    VERSION = tfds.core.Version("1.1.0" if ACTION_MODE == "delta" else "1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release for RoboTwin place_object dataset.",
        "1.1.0": "Delta joint action labels with absolute proprio state.",
    }

    N_WORKERS = 32
    MAX_PATHS_IN_MEMORY = 100
    PARSE_FCN = _generate_examples

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                    ),
                                    "left_wrist_image": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                    ),
                                    "right_wrist_image": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                    ),
                                    "low_cam_image": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(14,), dtype=np.float32
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(14,), dtype=np.float32
                            ),
                            "discount": tfds.features.Scalar(dtype=np.float32),
                            "reward": tfds.features.Scalar(dtype=np.float32),
                            "is_first": tfds.features.Scalar(dtype=np.bool_),
                            "is_last": tfds.features.Scalar(dtype=np.bool_),
                            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                            "language_instruction": tfds.features.Sequence(
                                tfds.features.Text()
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(),
                        }
                    ),
                }
            )
        )

    def _split_paths(self):
        train_files = glob.glob(
            "processed_data/stack_blocks_two/train/*.hdf5"
        )+glob.glob(
            "processed_data/stack_blocks_three/train/*.hdf5"
        )+glob.glob(
            "processed_data/stack_bowls_two/train/*.hdf5"
        )+glob.glob(
            "processed_data/stack_bowls_three/train/*.hdf5"
        )
        val_files = glob.glob(
            "processed_data/stack_blocks_two/val/*.hdf5"
        )+glob.glob(
            "processed_data/stack_blocks_three/val/*.hdf5"
        )+glob.glob(
            "processed_data/stack_bowls_two/val/*.hdf5"
        )+glob.glob(
            "processed_data/stack_bowls_three/val/*.hdf5"
        )

        print(f"[INFO] Found {len(train_files)} training files")
        print(f"[INFO] Found {len(val_files)} validation files")

        return {
            "train": train_files,
            "val": val_files,
        }


if __name__ == "__main__":
    builder = robotwin4stack_rand_aloha()
    builder.download_and_prepare()

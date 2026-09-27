"""Arrange RGB blocks according to the episode's language-conditioned goal."""
from copy import deepcopy

import numpy as np

from .blocks_ranking_rgb import blocks_ranking_rgb
from .color_order_goal import COLORS, ORDERS, episode_info, resolve_order, row_success
from .utils import create_box, rand_pose


class blocks_ranking_color_order(blocks_ranking_rgb):
    # Only the existing pick-and-place primitive is reused. Scene sampling,
    # target assignment, language metadata and success semantics are defined here.
    def setup_demo(self, **kwargs):
        self.scene_seed = int(kwargs.get('seed', 0))
        self.target_order = resolve_order(kwargs.get('target_order', 'random'), self.scene_seed)
        super().setup_demo(**kwargs)
        # Base_Task initializes self.info AFTER load_actors; set metadata here.
        self.info.update(episode_info(self.target_order, self.scene_seed))

    def load_actors(self):
        self.block_half_size = float(np.random.uniform(0.015, 0.025))
        table_z = 0.74 + self.table_z_bias
        while True:
            poses = []
            for _ in range(3):
                while True:
                    pose = rand_pose(
                        xlim=[-0.28, 0.28], ylim=[-0.08, 0.05],
                        zlim=[0.74 + self.block_half_size], qpos=[1, 0, 0, 0],
                        ylim_prop=True, rotate_rand=True, rotate_lim=[0, 0, 0.75],
                    )
                    if abs(pose.p[0]) < 0.05:
                        continue
                    if np.sum((pose.p[:2] - np.array([0, -0.1])) ** 2) < 0.01:
                        continue
                    if any(np.sum((pose.p[:2] - other.p[:2]) ** 2) < 0.01 for other in poses):
                        continue
                    poses.append(deepcopy(pose))
                    break
            positions = {c: pose.p for c, pose in zip('RGB', poses)}
            # The initial scene is independent of the requested order and is
            # never already a valid completed row for any of the six goals.
            if not any(row_success(positions, o, self.block_half_size, 0.74) for o in ORDERS):
                break

        self.blocks = {}
        for color, pose, rgb in zip('RGB', poses, ((1, 0, 0), (0, 1, 0), (0, 0, 1))):
            block = create_box(self, pose=pose, half_size=(self.block_half_size,) * 3,
                               color=rgb, name=f'{COLORS[color]}_block')
            self.blocks[color] = block
            self.add_prohibit_area(block, padding=0.05)
        self.prohibited_area.append([-0.17, -0.22, 0.17, -0.12])

        y = np.random.uniform(-0.2, -0.1)
        self.target_slots = [
            [np.random.uniform(low, high), y, table_z, 0, 1, 0, 0]
            for low, high in ((-0.09, -0.08), (-0.01, 0.01), (0.08, 0.09))
        ]

    def play_once(self):
        self.last_gripper = None
        for color, target in zip(self.target_order, self.target_slots):
            self.pick_and_place_block(self.blocks[color], target)
        return self.info

    def check_success(self):
        return row_success(
            {c: block.get_pose().p for c, block in self.blocks.items()},
            self.target_order, self.block_half_size, 0.74 + self.table_z_bias,
            grippers_open=self.is_left_gripper_open() and self.is_right_gripper_open(),
        )

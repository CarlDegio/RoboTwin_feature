"""Pipeline tests run in the RoboTwin environment; no GPU scene is created."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from scripts.collect_data import build_episode_instructions, run
from scripts.eval_policy_xpolicylab import active_episode_info, build_instruction
from envs.color_order_goal import episode_info, TASK_NAME


class PipelineTests(unittest.TestCase):
    def test_failed_replay_is_not_published_as_a_training_episode(self):
        env = Mock()
        env.info = episode_info('RGB', 0)
        env.play_once.return_value = env.info
        env.check_success.return_value = False
        env.load_tran_data.return_value = {
            'task_goal': env.info['task_goal'], 'left_joint_path': [], 'right_joint_path': [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'seed.txt').write_text('0')
            with self.assertRaisesRegex(RuntimeError, 'replay'):
                run(env, {
                    'task_name': TASK_NAME, 'save_path': tmp, 'episode_num': 1,
                    'use_seed': True, 'collect_data': True, 'clear_cache_freq': 5,
                })
            env.merge_pkl_to_hdf5_video.assert_not_called()
            self.assertFalse((Path(tmp) / 'instruction/episode_0000000.json').exists())

    def test_eval_uses_live_goal_without_expert_check(self):
        live = episode_info('BGR', 42)
        actual = active_episode_info({'task_name': TASK_NAME}, SimpleNamespace(info=live), {'info': {}})
        self.assertEqual(actual, live)

    def test_eval_reset_rejects_a_different_goal_from_expert_check(self):
        with self.assertRaises(ValueError):
            active_episode_info(
                {'task_name': TASK_NAME}, SimpleNamespace(info=episode_info('RGB', 42)),
                episode_info('BGR', 42),
            )

    def test_eval_cannot_silently_replace_missing_goal_with_task_name(self):
        with self.assertRaises(ValueError):
            build_instruction({'task_name': TASK_NAME}, {'info': {}}, 'seen', 1)

    def test_eval_rejects_instruction_mapping_that_disagrees_with_goal(self):
        info = episode_info('BGR', 42)
        info['info']['{A}'] = 'red block'
        with self.assertRaises(ValueError):
            build_instruction({'task_name': TASK_NAME}, info, 'seen', 1)

    def test_collection_cannot_save_a_goal_free_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                build_episode_instructions({'task_name': TASK_NAME, 'save_path': tmp}, {'info': {}}, 0)
            self.assertFalse((Path(tmp) / 'instruction/episode_0000000.json').exists())

    def test_initialization_metadata_is_sufficient_without_expert_rollout(self):
        info = episode_info('GBR', 42)
        text = build_instruction({'task_name': TASK_NAME}, info, 'unseen', 10)
        self.assertLess(text.index('green block'), text.index('blue block'))
        self.assertLess(text.index('blue block'), text.index('red block'))

    def test_hdf5_contains_goal_alongside_language(self):
        import h5py
        import numpy as np
        from envs.utils.pkl2hdf5 import create_xpolicylab_hdf5
        info = episode_info('BRG', 42)
        data = {'joint_action': {'left_arm': np.zeros((2, 6))}, 'observation': {}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'episode.hdf5'
            try:
                create_xpolicylab_hdf5(data, path, ['blue, red, green'], 15, task_goal=info['task_goal'])
            except TypeError as exc:
                self.fail(f'HDF5 writer does not accept task goal metadata: {exc}')
            with h5py.File(path) as f:
                self.assertEqual(json.loads(f['additional_info/task_goal'].asstr()[()]), info['task_goal'])
                self.assertEqual(json.loads(f['instructions'].asstr()[()]), ['blue, red, green'])


if __name__ == '__main__':
    unittest.main()

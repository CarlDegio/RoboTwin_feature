"""Opt-in SAPIEN checks: COLOR_ORDER_SIM_TESTS=1 python -m unittest discover -s tests."""
import os
import unittest


@unittest.skipUnless(os.environ.get('COLOR_ORDER_SIM_TESTS') == '1', 'requires simulator assets and a GPU')
class ColorOrderSimulationTests(unittest.TestCase):
    def test_goals_share_the_same_scene_and_survive_eval_reset(self):
        import numpy as np
        from envs.blocks_ranking_color_order import blocks_ranking_color_order
        from envs.color_order_goal import ORDERS, TASK_NAME, episode_info
        from scripts.eval_policy_xpolicylab import active_episode_info, build_instruction, load_task_args

        args, _ = load_task_args({
            'task_name': TASK_NAME, 'task_config': 'color_order', 'policy_name': 'validation',
        })
        args.update(save_data=False, eval_mode=True, render_freq=0)
        task = blocks_ranking_color_order()
        reference = None
        for index, order in enumerate((*ORDERS, 'random', 'random')):
            with self.subTest(order=order, episode=index):
                args['target_order'] = order
                task.setup_demo(now_ep_num=index, seed=42, **args)
                try:
                    snapshot = np.concatenate([
                        *[np.r_[task.blocks[c].get_pose().p, task.blocks[c].get_pose().q] for c in 'RGB'],
                        np.asarray(task.target_slots).ravel(), [task.block_half_size],
                    ])
                    if reference is None:
                        reference = snapshot
                    else:
                        np.testing.assert_allclose(snapshot, reference, atol=1e-6, rtol=0)
                    self.assertFalse(task.check_success())
                    expected = episode_info(order, 42)
                    live = active_episode_info(args, task, expected)
                    self.assertEqual(live['task_goal'], expected['task_goal'])
                    text = build_instruction(args, live, 'unseen', 6)
                    colors = live['task_goal']['target_order']
                    self.assertLess(text.index(colors[0] + ' block'), text.index(colors[1] + ' block'))
                    self.assertLess(text.index(colors[1] + ' block'), text.index(colors[2] + ' block'))
                finally:
                    task.close_env()


if __name__ == '__main__':
    unittest.main()

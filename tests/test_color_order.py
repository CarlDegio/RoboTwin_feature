"""Goal semantics can be tested without importing SAPIEN or allocating a GPU."""
import importlib
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "envs"))
sys.path.insert(0, str(ROOT))


class ColorOrderTests(unittest.TestCase):
    def setUp(self):
        try:
            self.goal = importlib.import_module("color_order_goal")
        except ModuleNotFoundError:
            self.fail("The color-order goal implementation is missing")

    def test_all_orders_accept_only_the_requested_final_arrangement(self):
        orders = ("RGB", "RBG", "GRB", "GBR", "BRG", "BGR")
        for actual in orders:
            positions = {c: (x, -0.15, 0.76) for c, x in zip(actual, (-0.085, 0, 0.085))}
            for requested in orders:
                with self.subTest(actual=actual, requested=requested):
                    self.assertEqual(self.goal.row_success(positions, requested, 0.02, 0.74), actual == requested)

    def test_rejects_overlap_stacking_nonrow_and_nonfinite_positions(self):
        valid = {"R": (-0.085, -0.15, 0.76), "G": (0, -0.15, 0.76), "B": (0.085, -0.15, 0.76)}
        for bad_green in ((-0.08, -0.15, 0.76), (0, -0.15, 0.8), (0, 0, 0.76), (float('nan'), -0.15, 0.76)):
            with self.subTest(position=bad_green):
                self.assertFalse(self.goal.row_success({**valid, "G": bad_green}, "RGB", 0.02, 0.74))
        self.assertFalse(self.goal.row_success(valid, "RGB", 0.02, 0.74, grippers_open=False))

    def test_table_height_and_block_size_are_used_in_success(self):
        positions = {c: (x, -0.15, 0.705) for c, x in zip('BRG', (-0.085, 0, 0.085))}
        self.assertTrue(self.goal.row_success(positions, 'BRG', 0.025, 0.68))
        self.assertFalse(self.goal.row_success(positions, 'BRG', 0.025, 0.74))

    def test_goal_is_reproducible_without_consuming_global_random_state(self):
        random.seed(91)
        before = random.getstate()
        first = self.goal.resolve_order('random', 57)
        for _ in range(10):
            self.assertEqual(self.goal.resolve_order('random', 57), first)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(self.goal.resolve_order('BRG', 999), 'BRG')
        self.assertEqual(len({self.goal.resolve_order('random', i) for i in range(100)}), 6)
        for invalid in ('RRB', 'XYZ', '', None):
            with self.assertRaises(ValueError):
                self.goal.resolve_order(invalid, 0)

    def test_balancing_uses_successful_episode_index_and_fixed_order_wins(self):
        expected = ['RGB', 'RBG', 'GRB', 'GBR', 'BRG', 'BGR'] * 2
        actual = [self.goal.collection_order('random', i, balanced=True) for i in range(12)]
        self.assertEqual(actual, expected)
        self.assertEqual(self.goal.collection_order('GBR', 9, balanced=True), 'GBR')

    def test_all_instruction_variants_describe_the_same_goal(self):
        from description.utils.generate_episode_instructions import generate_episode_descriptions
        for order in ('RGB', 'RBG', 'GRB', 'GBR', 'BRG', 'BGR'):
            info = self.goal.episode_info(order, 8)
            generated = generate_episode_descriptions('blocks_ranking_color_order', [info['info']], 20)
            self.assertTrue(generated)
            for split in ('seen', 'unseen'):
                self.assertTrue(generated[0][split])
                for text in generated[0][split]:
                    colors = [self.goal.COLORS[c] + " block" for c in order]
                    self.assertLess(text.index(colors[0]), text.index(colors[1]))
                    self.assertLess(text.index(colors[1]), text.index(colors[2]))
                    self.assertNotIn('{', text)
            self.assertEqual(info['task_goal']['order_id'], order)
            self.assertEqual(info['task_goal']['scene_seed'], 8)

    def test_summary_keeps_missing_orders_and_reports_macro_average(self):
        records = [dict(order_id='RGB', success=True), dict(order_id='RGB', success=False), dict(order_id='BGR', success=True)]
        summary = self.goal.summarize_results(records)
        self.assertEqual(summary['by_order']['RGB']['success_rate'], 0.5)
        self.assertEqual(summary['by_order']['BGR']['success_rate'], 1.0)
        self.assertIsNone(summary['by_order']['GBR']['success_rate'])
        self.assertIsNone(summary['macro_success_rate'])
        records.extend(dict(order_id=o, success=False) for o in ('RBG', 'GRB', 'GBR', 'BRG'))
        self.assertEqual(self.goal.summarize_results(records)['macro_success_rate'], 0.25)


if __name__ == '__main__':
    unittest.main()

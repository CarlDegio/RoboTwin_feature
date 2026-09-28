import argparse
import unittest

from scripts.eval_pi05_color_order import build_commands


class Pi05EvalSplitTests(unittest.TestCase):
    def args(self, split):
        return argparse.Namespace(split=split, scheduler_config="config.yml", checkpoint="checkpoint",
                                  eval_env="RoboTwin", test_num=100, seed=3, instruction_type="seen", dry_run=True)

    def test_iid_and_ood_jobs_are_disjoint_and_equally_sized(self):
        iid = build_commands(self.args("iid"))
        ood = build_commands(self.args("ood"))
        self.assertEqual([order for order, _ in iid], ["RBG", "GRB", "GBR", "BRG", "BGR"])
        self.assertEqual([order for order, _ in ood], ["RGB"])
        for _, command in iid + ood:
            self.assertEqual(command[command.index("--test-num") + 1], "100")
            self.assertEqual(command[command.index("--seed") + 1], "3")
            self.assertIn("--no-expert-check", command)
            self.assertIn("--dry-run", command)


if __name__ == "__main__":
    unittest.main()

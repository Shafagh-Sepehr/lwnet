import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.checkpoint_selection import compare_checkpoint, parse_metric_policy


class CheckpointSelectionTests(unittest.TestCase):
    def test_single_metric_is_strict(self):
        policy = parse_metric_policy(' auc ')
        self.assertEqual(policy['metric_order'], ['auc'])
        self.assertEqual(policy['metric_tolerances'], {'auc': 0.0})
        self.assertEqual(compare_checkpoint({'auc': 0.5}, {'auc': 0.5}, ['auc'], {'auc': 0.0})[1], 'all_tied')

    def test_ordered_near_tie_selects_by_dice(self):
        policy = parse_metric_policy('auc,dice,loss')
        selected, reason, details = compare_checkpoint(
            {'auc': 0.984430, 'dice': 0.8069, 'loss': 0.1878},
            {'auc': 0.984440, 'dice': 0.8037, 'loss': 0.1921},
            policy['metric_order'], policy['metric_tolerances'])
        self.assertTrue(selected)
        self.assertEqual(reason, 'dice')
        self.assertEqual(details['tie_chain'], ['auc'])

    def test_loss_is_minimized_and_boundary_is_decisive(self):
        self.assertTrue(compare_checkpoint({'loss': 1.0}, {'loss': 1.1}, ['loss'], {'loss': 0.1})[0])
        self.assertFalse(compare_checkpoint({'loss': 1.0}, {'loss': 1.1}, ['loss'], {'loss': 0.1000001})[0])

    def test_invalid_policy_and_values(self):
        with self.assertRaises(ValueError):
            parse_metric_policy('auc,auc')
        with self.assertRaises(ValueError):
            parse_metric_policy('tr_auc,auc')
        with self.assertRaises(ValueError):
            parse_metric_policy('auc', 'auc=nan')
        self.assertEqual(compare_checkpoint({'auc': math.nan}, None, ['auc'], {'auc': 0})[1], 'invalid_candidate')


if __name__ == '__main__':
    unittest.main()

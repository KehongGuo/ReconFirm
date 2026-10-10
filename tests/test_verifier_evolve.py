"""Rungs 2 and 3 of the self-evolving verifier, without a model server."""
import unittest

import verifier_evolve as ve  # noqa: E402


def rec(real, answer, gt=None, target='plant'):
    return {'id': 'r/1', 'target': target, 'track_was_real': real, 'gt_top': gt,
            'answer': {'category': answer}, 'prompt': 'Q', 'vocabulary': ['plant', 'vase', 'chair']}


class TruthTests(unittest.TestCase):
    def test_truth_names_annotation_without_structure(self):
        t = ve.truth_text(rec(False, 'plant', [['vase', 900], ['wall', 500]]))
        self.assertIn('not a plant', t)
        self.assertIn('vase', t)
        self.assertNotIn('wall', t)

    def test_truth_real(self):
        self.assertEqual(ve.truth_text(rec(True, 'vase')), 'the centre object is a plant')


class SelfCorrectionTests(unittest.TestCase):
    def test_keeps_only_right_corrections(self):
        replies = iter([('CATEGORY: vase\nCLEAR: yes\nDESCRIPTION: dry twigs in a vase', {'category': 'vase', 'clear': True, 'description': 'dry twigs in a vase'}),
                        ('CATEGORY: plant\nCLEAR: no\nDESCRIPTION: a plant', {'category': 'plant', 'clear': False, 'description': 'a plant'})])
        ve.ask, orig = (lambda *a, **k: next(replies)), ve.ask
        try:
            kept, failed = ve.self_corrections([rec(False, 'plant'), rec(False, 'plant')], 'm', 0)
        finally:
            ve.ask = orig
        self.assertEqual([r['category'] for r in kept], ['vase'])
        self.assertEqual(len(failed), 1)


class MutateTests(unittest.TestCase):
    def test_variant_must_keep_truth_once(self):
        for reply, ok in [('Correct it. Truth: {truth}.', True), ('Correct it.', False),
                          ('{truth} and {truth}', False), ('Truth {truth} {other}', False)]:
            ve.chat_text, orig = (lambda *a, **k: reply), ve.chat_text
            try:
                self.assertEqual(ve.mutate('M', 'P', 'm', 0, seed=0) is not None, ok, reply)
            finally:
                ve.chat_text = orig


if __name__ == '__main__':
    unittest.main()

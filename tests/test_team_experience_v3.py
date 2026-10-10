import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import team_agents as ag
import team_experience_v3 as v3
import team_run


def run_dir(root, name, scene, target, real, dog_text, drone_text='a drone view'):
    d = Path(root) / name
    d.mkdir()
    (d / 'manifest.json').write_text(json.dumps({'episode': {'episode_id': f'{scene}:0', 'scene': scene,
                                                             'target': target}}))
    ev = [{'t': 1, 'event': 'verify_reply', 'robot': 'd0', 'track': 'T1',
           'answer': {'category': target, 'clear': True, 'description': drone_text}},
          {'t': 2, 'event': 'verify_reply', 'robot': 'dog0', 'track': 'T1',
           'answer': {'category': target, 'clear': True, 'description': dog_text}},
          {'t': 2, 'event': 'candidate_decision', 'robot': 'dog0', 'track': 'T1',
           'record': {'p_target': 0.8}, 'context': {'near_labels': ['window']}},
          {'t': 3, 'event': 'declare', 'track': 'T1', 'track_was_real': real, 'success': real}]
    (d / 'events.jsonl').write_text('\n'.join(json.dumps(e) for e in ev))
    return d


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def store(self):
        s = v3.new_store('test')
        for k, (real, text) in enumerate([(True, 'a green potted plant with broad leaves'),
                                          (True, 'a leafy green plant in a pot'),
                                          (False, 'a brown flower vase holding dried branches')]):
            v3.add(s, v3.harvest(run_dir(self.root, f'r{k}', f'S{k}', 'plant', real, text)))
        return s

    def test_harvest_keeps_the_dogs_own_view_and_the_outcome_bit(self):
        c = v3.harvest(run_dir(self.root, 'r', 'S', 'plant', False, 'a vase', drone_text='a plant'))
        self.assertEqual(c['descriptions'], ['a vase'])
        self.assertFalse(c['accepted'])
        self.assertEqual(c['near'], ['window'])

    def test_retrieval_is_balanced_and_most_similar_first(self):
        store = self.store()
        acc, ref = v3.similar(store, 'plant', {'descriptions': ['a vase with dried branches']})
        self.assertEqual((len(acc), len(ref)), (1, 1))      # 2 accepted, 1 refused -> one of each
        self.assertEqual(v3.similar(store, 'toilet', {'descriptions': ['a toilet']}), ([], []))

    def test_protocol_guards(self):
        path = Path(self.root) / 'store.json'
        s = self.store()
        v3.save(s, path)
        with self.assertRaises(SystemExit):     # learning on the evaluation set
            v3.install(path, {'episode_id': 'X:0', 'scene': 'X', 'role': 'hm3d_val'}, learn=True)
        with self.assertRaises(SystemExit):     # testing on a scene the memory was built on
            v3.install(path, {'episode_id': 'S0:5', 'scene': 'S0', 'role': 'hm3d_val'})
        v3.save({**s, 'frozen': True}, path)
        with self.assertRaises(SystemExit):     # a frozen store is read-only
            v3.install(path, {'episode_id': 'X:0', 'scene': 'X', 'role': 'hm3d_train'}, learn=True)


class VetoTests(unittest.TestCase):
    def setUp(self):
        self.saved = (ag.JevDecider.candidate, team_run.Runner.run)
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'store.json'
        s = v3.new_store('test')
        for k, (real, text) in enumerate([(True, 'a green potted plant'), (False, 'a flower vase')]):
            v3.add(s, v3.harvest(run_dir(self.tmp.name, f'r{k}', f'S{k}', 'plant', real, text)))
        v3.save(s, self.path)
        self.stub = lambda self, role, name, track, ctx, now: (
            'declare_found' if role == 'dog' else 'request_dog', 0.5, {'p_target': 0.9})

    def tearDown(self):
        ag.JevDecider.candidate, team_run.Runner.run = self.saved
        self.tmp.cleanup()

    def decide(self, p_owner, role='dog', track_id='T1', learn=False):
        ag.JevDecider.candidate = self.stub        # one install per process, as in a run
        v3.install(self.path, {'episode_id': 'X:0', 'scene': 'X', 'role': 'hm3d_train'}, learn=learn)
        calls = []
        client = SimpleNamespace(judge=lambda state, q: calls.append(state) or
                                 {'owner_accepts': {'value': p_owner}})
        track = SimpleNamespace(track_id=track_id, vlm=[{'by': 'dog0', 'category': 'plant', 'clear': True,
                                                         'description': 'a vase with a plant'}])
        out = ag.JevDecider.candidate(SimpleNamespace(client=client), role, 'dog0', track,
                                      {'target': 'plant', 'near_labels': []}, 0.0)
        return out, calls

    def test_the_owner_would_refuse_it_the_dog_rejects_instead_of_declaring(self):
        (action, latency, rec), calls = self.decide(0.1)
        self.assertEqual(action, 'reject')
        self.assertTrue(rec['rsi']['veto'])
        self.assertEqual(len(calls), 1)
        self.assertNotIn('rate', json.dumps(calls[0]))          # no base rate in the boundary state
        self.assertGreaterEqual(latency, 0.5)                    # the extra call is charged on top

    def test_the_owner_would_accept_it_the_declaration_stands(self):
        (action, _, rec), _ = self.decide(0.9)
        self.assertEqual(action, 'declare_found')
        self.assertNotIn('veto', rec['rsi'])

    def test_drones_are_untouched_and_cost_nothing(self):
        (action, _, _), calls = self.decide(0.1, role='drone')
        self.assertEqual((action, calls), ('request_dog', []))

    def test_learning_sometimes_declares_anyway_to_label_a_veto(self):
        tid = next(t for t in (f'T{i}' for i in range(100)) if v3.explored('X:0', t))
        (action, _, rec), _ = self.decide(0.1, track_id=tid, learn=True)
        self.assertEqual(action, 'declare_found')
        self.assertTrue(rec['rsi']['explore'])
        (action, _, _), _ = self.decide(0.1, track_id=tid)       # frozen use never explores
        self.assertEqual(action, 'reject')

    def test_learning_appends_the_finished_run_unless_it_had_service_errors(self):
        run = run_dir(self.tmp.name, 'new', 'X', 'plant', True, 'a small plant')
        for errors, expected in ((1, 2), (0, 3)):
            team_run.Runner.run = lambda self, e=errors: {'counts': {'service_errors': e}}
            ag.JevDecider.candidate = self.stub
            v3.install(self.path, {'episode_id': 'X:0', 'scene': 'X', 'role': 'hm3d_train'}, learn=True)
            team_run.Runner.run(SimpleNamespace(out=run))
            self.assertEqual(len(json.loads(self.path.read_text())['cases']), expected)
        self.assertFalse([f for f in os.listdir(self.tmp.name) if '.tmp' in f])


if __name__ == '__main__':
    unittest.main()

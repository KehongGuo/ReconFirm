import math
import unittest
from types import SimpleNamespace

import numpy as np

import team_agents as ag
from aerial_map import AerialMap


def room_with_table():
    """8 x 8 m room; a 1 m table (top 0.75 m) in the middle; a 1.9 m wardrobe."""
    m = AerialMap((4.0, 4.0), floor_y=0.0, size_m=10, res=0.1)
    ii, jj = np.mgrid[0:m.n, 0:m.n]
    wx, wz = m.world(ii, jj)
    inside = (wx > 0.5) & (wx < 7.5) & (wz > 0.5) & (wz < 7.5)
    m.free_hits[inside] = 1
    m.seen[inside] = True
    table = (np.abs(wx - 4.0) < 0.5) & (wz > 0.5) & (wz < 7.5)     # a wall-to-wall low table
    m.occ_hits[table] = 1                                           # blocks the dog
    wardrobe = (np.abs(wx - 2.0) < 0.3) & (np.abs(wz - 4.0) < 0.3)
    m.occ_hits[wardrobe] = 1
    m.occ_air[wardrobe] = 1                                         # blocks everyone
    m._cache = None
    return m


class TwoLayerMapTests(unittest.TestCase):
    def test_a_drone_flies_over_furniture_the_dog_must_walk_round(self):
        m = room_with_table()
        dog = m.distance_field(2.0, 2.0, 'ground')
        air = m.distance_field(2.0, 2.0, 'air')
        gi, gj = m.cell(6.0, 2.0)
        self.assertFalse(np.isfinite(m.dist_at(dog, np.array([gi]), np.array([gj]))[0]),
                         'the table spans the room, so the dog cannot cross')
        self.assertTrue(np.isfinite(m.dist_at(air, np.array([gi]), np.array([gj]))[0]))

    def test_tall_obstacles_block_the_drone_too(self):
        m = room_with_table()
        air = m.distance_field(1.0, 4.0, 'air')
        path = m.path_to(air, *m.cell(3.0, 4.0))
        self.assertTrue(path)
        self.assertFalse(any(abs(x - 2.0) < 0.3 and abs(z - 4.0) < 0.3 for x, z in path))

    def test_unseen_space_is_closed_to_both_layers(self):
        m = AerialMap((4.0, 4.0), floor_y=0.0, size_m=10, res=0.1)
        ii, jj = np.mgrid[0:m.n, 0:m.n]
        wx, wz = m.world(ii, jj)
        pocket = (wx > 1) & (wx < 2) & (wz > 1) & (wz < 2)
        m.free_hits[pocket] = 1
        m.seen[pocket] = True
        m._cache = None
        for layer in ('ground', 'air'):
            f = m.distance_field(1.5, 1.5, layer)
            self.assertEqual(m.path_to(f, *m.cell(6.0, 6.0)), [], layer)


class SearchFrontierTests(unittest.TestCase):
    def test_space_seen_only_from_afar_is_still_a_frontier(self):
        # The whole room is known but only its left half was looked at from
        # close range: the edge of the looked-at half is a frontier.
        m = AerialMap((4.0, 4.0), floor_y=0.0, size_m=10, res=0.1)
        ii, jj = np.mgrid[0:m.n, 0:m.n]
        wx, wz = m.world(ii, jj)
        inside = (wx > 0.5) & (wx < 7.5) & (wz > 0.5) & (wz < 7.5)
        wall = ((wx > 0.3) & (wx < 7.7) & (wz > 0.3) & (wz < 7.7)) & ~inside
        m.free_hits[inside] = 1
        m.occ_hits[wall] = 1
        m.occ_air[wall] = 1
        m.seen[inside | wall] = True
        m.looked[(inside | wall) & (wx < 3.0)] = True
        m._cache = None
        fr = m.frontiers()
        self.assertTrue(fr)
        self.assertTrue(all(2.5 < f['centroid'][0] < 3.1 for f in fr), [f['centroid'] for f in fr])
        m.looked[inside | wall] = True
        m._cache = None
        self.assertEqual(m.frontiers(), [])


def world(drone_xz=(0.0, 0.0), dog_xz=(10.0, 0.0), battery=100.0):
    d = SimpleNamespace(aerial=True, state='airborne', battery=battery, x=drone_xz[0], z=drone_xz[1])
    g = SimpleNamespace(aerial=False, state='ground', battery=math.inf, x=dog_xz[0], z=dog_xz[1])
    return SimpleNamespace(bodies={'d0': d, 'dog0': g}, dog=g)


class BatteryTests(unittest.TestCase):
    def test_recall_fires_when_the_reserve_is_reached(self):
        # 10 m from the dog at 2 m/s with a 1.3 detour factor is 6.5 s home.
        self.assertFalse(ag.needs_recharge(world(battery=40.0), 'd0'))
        self.assertTrue(ag.needs_recharge(world(battery=6.5 + ag.RESERVE_S - 0.1), 'd0'))

    def test_a_trip_is_affordable_only_with_the_way_home_included(self):
        w = world(drone_xz=(0.0, 0.0), dog_xz=(0.0, 0.0), battery=30.0)
        self.assertTrue(ag.affordable(w, 'd0', (5.0, 0.0)))
        self.assertFalse(ag.affordable(w, 'd0', (20.0, 0.0)))

    def test_the_dog_never_needs_a_recharge(self):
        self.assertFalse(ag.needs_recharge(world(), 'dog0'))


def ctx(**over):
    base = {'target': 'toilet', 'distance_m': 0.8, 'path_m': 0.8, 'fresh_view': True,
            'handled_by': None, 'check_pending': False, 'dog_requested': False, 'near_labels': [],
            'task_desc': 'none', 'teammates': [], 'time_left_s': 100.0, 'battery_s': 80.0,
            'dog_distance_m': 6.0, 'can_declare': True}
    base.update(over)
    return base


def track(vlm=(), scores=(0.6,)):
    return SimpleNamespace(vlm=list(vlm), scores=list(scores), track_id='T1', label='toilet',
                           summary=lambda now: {'id': 'T1', 'label': 'toilet'})


class RoleOptionTests(unittest.TestCase):
    def test_drones_can_never_be_offered_a_declaration(self):
        opts = ag._options('drone', ctx(can_declare=True, distance_m=0.3), track())
        self.assertNotIn('declare_found', opts)
        self.assertIn('request_dog', opts)
        self.assertIn('inspect', opts)

    def test_the_dog_is_offered_a_declaration_only_within_range(self):
        self.assertIn('declare_found', ag._options('dog', ctx(distance_m=0.9), track()))
        self.assertNotIn('declare_found', ag._options('dog', ctx(distance_m=1.1), track()))
        self.assertNotIn('declare_found', ag._options('dog', ctx(distance_m=0.9, can_declare=False), track()))

    def test_a_large_object_counts_from_its_surface(self):
        # 1.5 m from a sofa's centre but 0.7 m from its near side.
        self.assertIn('declare_found', ag._options('dog', ctx(distance_m=1.5, reach_m=0.7), track()))
        self.assertNotIn('approach', ag._options('dog', ctx(distance_m=1.5, reach_m=0.7), track()))

    def test_the_dog_declares_only_after_its_own_camera_confirms(self):
        self.assertNotIn('declare_found', ag._options('dog', ctx(own_confirmation=False), track()))
        self.assertIn('declare_found', ag._options('dog', ctx(own_confirmation=True), track()))

    def test_the_dog_is_not_requested_twice(self):
        self.assertNotIn('request_dog', ag._options('drone', ctx(dog_requested=True), track()))


class GateTests(unittest.TestCase):
    def test_every_irreversible_or_costly_move_needs_the_probability(self):
        dog = {'declare_found': '', 'second_opinion': '', 'reject': '', 'continue_task': ''}
        drone = {'second_opinion': '', 'reject': '', 'request_dog': '', 'continue_task': '', 'inspect': ''}
        self.assertEqual(ag._gate('declare_found', 0.49, dog), 'second_opinion')
        self.assertEqual(ag._gate('reject', 0.31, drone), 'continue_task')
        self.assertEqual(ag._gate('reject', 0.29, drone), 'reject')
        self.assertEqual(ag._gate('request_dog', 0.4, drone), 'second_opinion')
        self.assertEqual(ag._gate('request_dog', 0.6, drone), 'request_dog')

    def test_a_team_wide_rejection_needs_evidence_against_the_candidate(self):
        drone = {'second_opinion': '', 'reject': '', 'request_dog': '', 'continue_task': '', 'inspect': ''}
        unclear = track([{'category': 'chair', 'clear': False}])
        clear = track([{'category': 'chair', 'clear': True}])
        self.assertEqual(ag._gate('reject', 0.09, drone, unclear, 'tv_monitor'), 'second_opinion')
        self.assertEqual(ag._gate('reject', 0.09, drone, clear, 'tv_monitor'), 'reject')

    def test_after_its_close_look_a_confident_drone_calls_the_dog(self):
        opts = ag._options('drone', ctx(distance_m=2.0, inspected=True), track())
        self.assertNotIn('inspect', opts)
        self.assertEqual(ag._gate('continue_task', 0.65, opts), 'request_dog')
        self.assertEqual(ag._gate('continue_task', 0.45, opts), 'continue_task')
        before = ag._options('drone', ctx(distance_m=2.0), track())
        self.assertEqual(ag._gate('inspect', 0.65, before), 'inspect')

    def test_an_unclear_own_look_needs_a_higher_probability_to_declare(self):
        dog = {'declare_found': '', 'second_opinion': '', 'reject': '', 'continue_task': ''}
        unclear = track([{'category': 'toilet', 'clear': False, 'by': 'dog0'}])
        clear = track([{'category': 'toilet', 'clear': True, 'by': 'dog0'}])
        self.assertEqual(ag._gate('continue_task', 0.6, dog, unclear, 'toilet'), 'continue_task')
        self.assertEqual(ag._gate('continue_task', 0.75, dog, unclear, 'toilet'), 'declare_found')
        self.assertEqual(ag._gate('continue_task', 0.6, dog, clear, 'toilet'), 'declare_found')

    def test_the_dog_walks_to_a_candidate_only_on_the_dispatch_probability(self):
        dog = ag._options('dog', ctx(distance_m=2.5, reach_m=2.0, fresh_view=False), track())
        self.assertIn('approach', dog)
        self.assertEqual(ag._gate('approach', 0.4, dog), 'continue_task')
        self.assertEqual(ag._gate('approach', 0.6, dog), 'approach')

    def test_in_range_the_probability_alone_decides_the_declaration(self):
        # The first debug run: P = 0.75 at 0.7 m and the model kept choosing to
        # "approach" for a minute. Code takes the irreversible step.
        dog = {'declare_found': '', 'second_opinion': '', 'reject': '', 'continue_task': ''}
        self.assertEqual(ag._gate('continue_task', 0.75, dog), 'declare_found')
        self.assertEqual(ag._gate('second_opinion', 0.51, dog), 'declare_found')
        self.assertEqual(ag._gate('continue_task', 0.45, dog), 'continue_task')

    def test_approach_is_not_offered_once_inside_declaring_range(self):
        self.assertNotIn('approach', ag._options('dog', ctx(distance_m=0.8), track()))
        self.assertIn('approach', ag._options('dog', ctx(distance_m=1.5), track()))


class RuleDeciderTests(unittest.TestCase):
    def test_a_drone_calls_the_dog_on_a_clear_matching_opinion(self):
        act, _, _ = ag.RuleDecider().candidate('drone', 'd0', track([{'category': 'toilet', 'clear': True}]),
                                               ctx(distance_m=3.0), 0)
        self.assertEqual(act, 'request_dog')

    def test_the_dog_declares_on_a_clear_matching_opinion_in_range(self):
        act, _, _ = ag.RuleDecider().candidate('dog', 'dog0', track([{'category': 'toilet', 'clear': True}]),
                                               ctx(), 0)
        self.assertEqual(act, 'declare_found')


class PlanParseTests(unittest.TestCase):
    def test_typed_tasks_are_parsed_and_unknown_types_dropped(self):
        text = ('```json\n{"assignments": [{"robot": "d0", "task": "explore", "target": "f3"},'
                '{"robot": "dog0", "task": "verify", "target": "t7"},'
                '{"robot": "d1", "task": "recharge", "target": null},'
                '{"robot": "d2", "task": "teleport", "target": "F1"}]}\n```')
        plan = ag.parse_plan(text)
        self.assertEqual([(a['robot'], a['task'], a['target']) for a in plan],
                         [('d0', 'explore', 'F3'), ('dog0', 'verify', 'T7'), ('d1', 'recharge', None)])

    def test_a_target_copied_with_its_distance_is_still_read(self):
        plan = ag.parse_plan('{"assignments": [{"robot": "d0", "task": "explore", "target": "F3 (7.0 m)"}]}')
        self.assertEqual(plan[0]['target'], 'F3')

    def test_unparseable_plans_are_none(self):
        self.assertIsNone(ag.parse_plan('I would send the drones north.'))




class MaskPlacementTests(unittest.TestCase):
    def test_the_object_is_placed_by_its_own_pixels_not_the_box_background(self):
        # A 2 m-deep object fills the left half of the box; a wall 5 m away fills the rest.
        from mask_localize import MaskLocalizer
        h, w = 100, 100
        depth = np.full((h, w), 5.0, np.float32)
        depth[40:60, 30:50] = 2.0
        mask = np.zeros((h, w), bool)
        mask[40:60, 30:50] = True
        box = (30, 40, 70, 60)
        eye, R = np.zeros(3), np.eye(3)
        pos, width, z = MaskLocalizer.place(mask, depth, eye, R, 90.0, box)
        self.assertAlmostEqual(z, 2.0, places=3)
        self.assertAlmostEqual(pos[2], -2.0, places=3)          # camera looks down -Z
        self.assertLess(width, 1.0)                              # 20 px at f=50 px and 2 m is 0.8 m
        box_median = float(np.median(depth[40:60, 30:70]))
        self.assertGreater(box_median, 3.0, 'the box median would have put it on the wall')

    def test_too_few_mask_pixels_gives_no_placement(self):
        from mask_localize import MaskLocalizer
        depth = np.full((50, 50), 2.0, np.float32)
        mask = np.zeros((50, 50), bool)
        mask[0:3, 0:3] = True
        self.assertIsNone(MaskLocalizer.place(mask, depth, np.zeros(3), np.eye(3), 90.0, (0, 0, 50, 50)))


class BlockedBodyTests(unittest.TestCase):
    def test_same_goal_clears_blocked(self):
        """A blocked dog given the same goal again must re-plan (HM3D val: 28 of 99 stuck)."""
        from types import SimpleNamespace
        from team_world import Body
        spec = SimpleNamespace(name='dog0', kind='ground', altitude=0.0, max_speed=1.0)
        b = Body(SimpleNamespace(floor_y=0.0), spec, 0.0, 0.0, 0.0)
        b.set_goal((2.0, 0.0), 0.0)
        b.status, b.path = 'blocked', [(1.0, 0.0)]
        b.set_goal((2.0, 0.0), 5.0)
        self.assertEqual(b.status, 'moving')
        self.assertEqual(b.path, [])


    def test_drone_escapes_to_nearest_flyable_cell(self):
        """A drone inside an obstacle's inflation flies back out first (hm3d_val5: 134 of 313)."""
        from types import SimpleNamespace
        import numpy as np
        from team_world import Body
        air = np.ones((20, 20), bool)
        air[5:15, 5:15] = False                    # the drone sits at (10, 10): not flyable
        m = SimpleNamespace(_derived=lambda: {'air': air}, cell=lambda x, z: (int(x), int(z)),
                            inside=lambda i, j: 0 <= i < 20 and 0 <= j < 20, world=lambda i, j: (float(i), float(j)))
        spec = SimpleNamespace(name='d0', kind='aerial', altitude=1.2, max_speed=2.0)
        b = Body(SimpleNamespace(floor_y=0.0, map=m), spec, 12.0, 10.0, 0.0)
        self.assertEqual(b._escape_cell(), (15.0, 10.0))
        b.x = 3.0
        self.assertIsNone(b._escape_cell())


class AuditFixTests(unittest.TestCase):
    def test_gate_returns_only_offered_actions(self):
        """A vetoed declare must not come out as an action the robot was not offered."""
        from team_agents import _gate
        from types import SimpleNamespace
        tr = SimpleNamespace(vlm=[{'by': 'dog0', 'category': 'toilet', 'clear': False}])
        act = _gate('declare_found', 0.6, ['continue_task', 'declare_found', 'reject'], tr, 'toilet')
        self.assertIn(act, ['continue_task', 'declare_found', 'reject'])

    def test_dog_clear_none_backs_a_reject(self):
        from team_agents import reject_backed
        from types import SimpleNamespace
        tr = SimpleNamespace(vlm=[{'by': 'dog0', 'category': 'none', 'clear': True}])
        self.assertTrue(reject_backed(tr, 'toilet'))
        tr = SimpleNamespace(vlm=[{'by': 'd0', 'category': 'none', 'clear': True}])
        self.assertFalse(reject_backed(tr, 'toilet'))

    def test_jev_retries_server_errors_not_client_errors(self):
        import jev
        c = jev.JevClient()
        calls = []

        def flaky(payload):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError('TypeSafe HTTP status 503')
            return {'answers': {}}
        c._post_once = flaky
        self.assertEqual(c._post({}), {'answers': {}})
        self.assertEqual(len(calls), 2)
        calls.clear()

        def no_credit(payload):
            calls.append(1)
            raise RuntimeError('TypeSafe HTTP status 402')
        c._post_once = no_credit
        with self.assertRaises(RuntimeError):
            c._post({})
        self.assertEqual(len(calls), 1)

    def test_service_errors_versus_bugs(self):
        from team_run import is_service_error
        self.assertTrue(is_service_error(TimeoutError('read timed out')))
        self.assertTrue(is_service_error(RuntimeError('TypeSafe HTTP status 520')))
        self.assertFalse(is_service_error(ValueError('Out of range float values are not JSON compliant')))

    def test_detector_exact_phrase_wins(self):
        from perception import OpenVocabDetector
        det = OpenVocabDetector.__new__(OpenVocabDetector)
        det.vocabulary, det.phrases = ['headboard', 'whiteboard', 'board'], ['headboard', 'whiteboard', 'board']
        self.assertEqual(det._label_from_phrase('board'), 'board')
        self.assertEqual(det._label_from_phrase('headboard'), 'headboard')




class PromptExampleTests(unittest.TestCase):
    def test_brain_examples_carry_no_real_ids(self):
        # the 8B brain copied ["F3", "F1"] / T4 from the example in 21-49% of replies
        import re
        for p in (ag.BRAIN_PROMPT, ag.BRAIN_PROMPT_HYBRID):
            tail = p[p.rindex('Answer with JSON'):]
            self.assertIsNone(re.search(r'"(F\d+|T\d+|d\d|r\d|dog0)"', tail))
        self.assertEqual(ag.parse_order('{"search_order": ["<frontier id>", "F2"]}'), ['F2'])


if __name__ == '__main__':
    unittest.main()


class FloorDepthTests(unittest.TestCase):
    def test_box_floor_pixels_do_not_pull_the_object_closer(self):
        """A box over an object 3 m away whose lower part is floor: the estimate is the object, not the floor."""
        import types
        import numpy as np
        from perception import VisualDetection, ground_detections
        h = w = 200; hfov = 90.0; fx = (w / 2) / np.tan(np.deg2rad(hfov) / 2); eye_h = 0.88
        depth = np.full((h, w), 3.0)
        for i in range(h // 2 + 1, h):                      # below the horizon: floor, z = eye_h * fx / (i - h/2)
            depth[i, :] = min(3.0, eye_h * fx / (i - h / 2))
        sim = types.SimpleNamespace(team=[types.SimpleNamespace(hfov=hfov)], index={'dog0': 0},
                                    camera_transform=lambda name: (np.array([0.0, eye_h, 0.0]), np.eye(3)))
        det = lambda: VisualDetection(label='sofa', score=0.5, box=(80, 60, 120, 199), pixels=40 * 139)
        near = ground_detections([det()], depth, sim, 'dog0')[0].distance
        fixed = ground_detections([det()], depth, sim, 'dog0', floor_y=0.0)[0].distance
        self.assertLess(near, 2.0)
        self.assertGreater(fixed, 2.8)


class TruncatedReplyTests(unittest.TestCase):
    def test_reply_cut_by_max_tokens_keeps_order_and_complete_assignments(self):
        import team_agents as ta
        cut = ('{"search_order": ["F3", "F1"], "assignments": [{"robot": "d0", "task": "inspect", "target": "T9", '
               '"why": "close"}, {"robot": "dog0", "task": "verify", "target": "T1", "why": "T1 is a high-conf')
        self.assertEqual(ta.parse_order(cut), ['F3', 'F1'])
        self.assertEqual([a['target'] for a in ta.parse_plan(cut)], ['T9'])
        self.assertIsNone(ta.parse_plan('no json here'))


class RejectBackedTests(unittest.TestCase):
    def test_contrary_evidence_must_name_another_class(self):
        import types
        from team_agents import reject_backed
        tr = lambda *a: types.SimpleNamespace(vlm=[{'category': c, 'clear': cl, 'by': 'd0'} for c, cl in a])
        self.assertFalse(reject_backed(tr(('bathtub', False), ('toilet', False)), 'toilet'))   # one names the target
        self.assertTrue(reject_backed(tr(('bathtub', False), ('sink', False)), 'toilet'))      # two other classes
        self.assertTrue(reject_backed(tr(('sink', True)), 'toilet'))                          # one clear other class
        self.assertFalse(reject_backed(tr(('none', False), ('none', False)), 'toilet'))


class DogHeadroomTests(unittest.TestCase):
    """The dog's obstacle band ends at its own headroom: an overhang it walks under is not a wall."""

    def test_overhang_above_headroom_is_not_a_dog_obstacle(self):
        import numpy as np
        from aerial_map import AerialMap
        depth = np.full((64, 64), 2.0, dtype=np.float32)      # a surface 2 m ahead...
        depth[32:] = 0.0                                       # ...seen only above the eye (0.8-1.2 m): a table top's edge
        eye, rot = np.array([0.0, 0.8, 0.0]), np.eye(3)        # camera 0.8 m above the floor, looking along -z
        tall = AerialMap((0.0, 0.0), floor_y=0.0, size_m=10.0)
        low = AerialMap((0.0, 0.0), floor_y=0.0, size_m=10.0, ground_top=0.6)
        for m in (tall, low):
            m.integrate(depth, eye, rot, 90.0, 0.0, stride=1)
        self.assertGreater(int(tall.occ_hits.sum()), 0)        # hits at 0.6-1.3 m block the dog with the old band
        self.assertEqual(int(low.occ_hits.sum()), 0)           # above the dog's headroom: not a wall for it


class AllocationTests(unittest.TestCase):
    def runner(self, order, tasks):
        import team_run
        r = object.__new__(team_run.Runner)
        fr = {f: {'id': f, 'size': 10, 'cost': {'d0': 2.0}, 'goal': {'d0': (0.0, 0.0)}} for f in ('F1', 'F2', 'F3')}
        r.registry = SimpleNamespace(active=fr)
        r.order, r.tasks, r.world = order, tasks, SimpleNamespace(map=SimpleNamespace(res=0.1))
        r.active = lambda n: True
        r.body = lambda n: SimpleNamespace(aerial=False, speed=1.0)
        r.usable = lambda n: fr
        r.frontier_task = lambda n, fid, src: {'kind': 'explore', 'target': fid}
        r.set_task = lambda n, t, why: tasks.__setitem__(n, t)
        return r

    def test_an_idle_drone_takes_a_free_frontier_not_the_top_ranked_one_another_drone_holds(self):
        tasks = {'d1': {'kind': 'explore', 'target': 'F1'}, 'd2': {'kind': 'explore', 'target': 'F2'}, 'd0': None}
        self.runner(['F1', 'F2', 'F3'], tasks).allocate(['d0'])
        self.assertEqual(tasks['d0']['target'], 'F3')

    def test_it_shares_only_when_every_frontier_is_held(self):
        tasks = {'d1': {'kind': 'explore', 'target': 'F1'}, 'd2': {'kind': 'explore', 'target': 'F2'},
                 'd3': {'kind': 'explore', 'target': 'F3'}, 'd0': None}
        self.runner(['F1', 'F2', 'F3'], tasks).allocate(['d0'])
        self.assertEqual(tasks['d0']['target'], 'F1')

    def test_pt_trades_how_likely_against_how_far(self):
        tasks = {'d0': None}
        r = self.runner([], tasks)
        r.registry.active['F1']['cost']['d0'] = 2.0       # next door, Jev: unlikely
        r.registry.active['F2']['cost']['d0'] = 8.0       # four times as far, Jev: likely
        r.registry.active['F3']['cost']['d0'] = 30.0
        r.pt, r.fp = True, {'F1': 0.1, 'F2': 0.8, 'F3': 0.0}
        r.allocate(['d0'])
        self.assertEqual(tasks['d0']['target'], 'F2')      # 8 / 0.8 = 10 s beats 2 / 0.1 = 20 s

    def test_pt_without_scores_is_nearest_first(self):
        tasks = {'d0': None}
        r = self.runner([], tasks)
        r.registry.active['F2']['cost']['d0'] = 1.0
        r.pt, r.fp = True, {}
        r.allocate(['d0'])
        self.assertEqual(tasks['d0']['target'], 'F2')


class ServiceErrorTests(unittest.TestCase):
    def test_a_jev_transport_failure_is_a_service_error_not_a_crash(self):
        import team_run
        self.assertTrue(team_run.is_service_error(RuntimeError('TypeSafe connection failed or timed out')))
        self.assertFalse(team_run.is_service_error(RuntimeError('Daily USD budget exhausted')))

"""Run one mission: three drones + one legged robot (the dog) searching an unknown building, under one condition.

Condition = allocation {geometric, brain, direct, jev} x drone decider {rule, jev, vlm} x dog decider {rule, jev, vlm}
x clock {charged, free} (named presets: configs/conditions.yaml; run.py builds this command line from a config).

Clock: model calls run while the world is paused and
their measured latency is charged on the mission clock; the VLM server holds
one request at a time, so brain plans, second opinions and VLM-local decisions
queue behind one another in simulated time. `--clock free` charges nothing,
reproducing step-budget evaluation in which the world waits for the model.

New bookkeeping: every dog `verify` trip is logged with where it started, how
far the dog walked, how it ended and whether the candidate was real (from the
evaluation sidecar, never shown to a policy). A trip that ends without a
correct declaration is an "empty run".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np  # noqa: E402

import tactical_board as tb  # noqa: E402
import team_agents as ag  # noqa: E402
import team_board as board  # noqa: E402
from search_agents import VLMServer, phrase, verify  # noqa: E402
from team_world import CONTROL_DT, DOCK_RADIUS, DOG, DRONE_ENDURANCE_S, DRONES, MISSION_S, TeamWorld  # noqa: E402

SPIN = {'drone': 2.1, 'dog': 3.2}
STAND_MIN_M, STAND_MAX_M = 0.6, 0.9   # dog standoff band from the object's own cells: 0.2-0.6 m put its 0.5 m
                                       # camera against a bed post, 8 looks "none" (Z6MFQCViBuw:v2s20)
MAX_REPOSITIONS = 4                    # per candidate: two standoffs alternated for 60 s (p53SfW6mjZe:12)
REPOSITION_S = 8.0                     # at a candidate this long with nothing to do: stand elsewhere
DOG_EXPLORES = True    # the dog searches too (--dog-explores no: it waits for a gated candidate; HM3D's homes)
NO_STANDOFF_S = 20.0                   # a requested candidate with no reachable standoff this long is dropped


def is_service_error(exc):
    """Network/HTTP trouble with a model service (retryable, counted); anything else -- a bad
    value such as inf in the request, a wrong type, an exhausted budget -- is a bug to surface."""
    import socket
    msg = str(exc)
    if 'Daily USD' in msg or 'budget' in msg.lower():
        return False
    return isinstance(exc, (TimeoutError, socket.timeout, ConnectionError, OSError)) or \
        (isinstance(exc, RuntimeError) and ('HTTP' in msg or 'status' in msg or
                                             # jev.py's transport failure: it crashed whole missions (MP3D dev
                                             # yqstnuAEVhm:v2s12 twice, a Jev frontier score at t = 0)
                                             'connection failed' in msg or 'timed out' in msg)) or \
        type(exc).__module__.startswith(('requests', 'urllib3', 'httpx'))
FIELD_EVERY = 10
BRAIN_FIRST_S = 2.5
BRAIN_PERIOD_S = 12.0
BRAIN_MIN_INTERVAL_S = 3.0
DECISION_COOLDOWN_S = 5.0
INSPECT_LIMIT_S = 20.0
NO_VIEW_GIVE_UP_S = 6.0
# Prompt tokens measured at about 1030 + 0.38 x board characters (dbg3 runs);
# the server's context is 2560 with 260 reserved for the reply, so the board
# must stay under ~3300 characters. A large house overflowed it at t = 56 s.
BOARD_CHAR_BUDGET = 2900
MAX_DOG_ATTEMPTS = 2      # verify trips per candidate the scheduler may order
DOG_WAIT_M = 4.0          # dog pauses while a recharging drone is this close
DOCK_ACCEPT_M = 2.0       # hybrid: the last 2 m onto the dog's back is a precision-landing step we do not model
LOOK_NEAR_M = 3.0         # en route the dog asks once a fresh view is this close
LOOK_EVERY_S = 3.0        # the dog at a candidate retries its own look this often
MEET_TIMEOUT_S = 40.0     # the dog gives up a rendezvous after this long
# Hybrid scheduler: the brain ranks frontiers, a Hungarian assignment does
# the geometry. Cost of sending robot r to frontier f, in seconds:
#   travel(r, f) / speed(r) + RANK_S * rank(f)   (+ DUP_S when it must share an already targeted f)
#   - SIZE_BONUS_M * min(width_m, 4) / speed(r)
RANK_S = 6.0
DUP_S = 8.0
# --frontier-cost pt: search theory's p/t rule instead of the rank penalty. Jev scores each frontier with a calibrated
# P(target beyond it) and a robot goes where travel time / P is smallest, so a likely exit far away and an unlikely one
# next door trade off by how likely and how far, not by a fixed 6 s per rank step.
P_FLOOR = 0.05                # a frontier Jev calls hopeless still costs a finite (20x) travel time
MIN_TRAVEL_S = 0.5            # the size bonus can take travel time to <= 0, where dividing by P would favour low P
SIZE_BONUS_M = 0.5
EXPLORE_HYSTERESIS_S = 8.0   # an explore task younger than this is not reassigned
HYBRID_BRAIN_PERIOD_S = 20.0
VERIFY_NEAR_M = 3.0          # the dog may verify a candidate this close without a request
FINAL_APPROACH_M = 0.6   # a confirmed dog farther than this from the object walks in before declaring
FINAL_STANDOFF_M = 0.45  # ... to this far from it
FINAL_APPROACH_S = 4.0   # ... for at most this long + 2 s per metre, then declares where it is
VIEW_AT_M = 0.3          # a dog's look is evidence for the spot it was taken from
FRESH_VIEW_S = 1.5        # must exceed the detection period (DETECT_EVERY x CONTROL_DT = 1.0 s)
BLACKLIST_S = 20.0
GEOMETRIC_DISPATCH_SCORE = 0.35
RECHARGE_BELOW_FRACTION = 0.5


class Runner:
    def __init__(self, episode, allocation, drone_local, dog_local, detector, server, jev,
                 out_dir=None, video=False, clock='charged', mission_s=MISSION_S, declare_roles=('dog',),
                 experience=None, localizer=None, localize_mode='box', scheduler='hybrid', frontier_cost='rank'):
        self.sched = scheduler
        self.pt = frontier_cost == 'pt'
        self.fp, self.fp_t = {}, -math.inf       # pt: the judge's P(target beyond frontier), when last asked
        self.order, self.order_t = [], -math.inf
        self._idle = []
        self._home = {}
        self.ep, self.allocation, self.clock, self.mission_s = episode, allocation, clock, mission_s
        # The case memory adds nothing to any decider's state: it checks a declaration against similar past
        # outcomes (team_experience_v3.install, from main).
        self.lessons = []
        self.world = TeamWorld(episode, detector=detector, declare_roles=declare_roles,
                               localizer=localizer, localize_mode=localize_mode)
        self.server = server
        # brain: VLM ranks frontiers (hybrid); direct: VLM assigns each robot a frontier itself
        # (the literature baseline); jev: no VLM scheduler, Jev ranks frontiers (J3).
        self.brain = (ag.Brain(server, hybrid=(scheduler == 'hybrid' and allocation == 'brain'))
                      if allocation in ('brain', 'direct') else None)
        self.jev_ranker = jev if allocation == 'jev' or self.pt else None
        self.rank_inflight = False
        self.dec = {'drone': ag.make_decider(drone_local, jev, server),
                    'dog': ag.make_decider(dog_local, jev, server)}
        from vocabulary import VOCABULARY
        self.vocabulary = VOCABULARY
        self.target = episode['target']
        self.registry = tb.FrontierRegistry()
        self.fields = {}
        self.names = self.world.names
        self.tasks = {n: None for n in self.names}
        self.spin_left = {n: SPIN['dog' if n == DOG else 'drone'] for n in self.names}
        self.pending, self._seq = [], 0
        self.brain_inflight, self.brain_last_reply, self.brain_wanted = False, -math.inf, False
        self.plan_note = None
        self._menu = {}
        self.inflight = {n: False for n in self.names}
        self.frontier_inflight = {n: False for n in self.names}
        self.cand, self.last_decision, self.arrived_at = {}, {}, {}
        self.blacklist = {n: {} for n in self.names}
        self.deferred = {}
        self.trips = []                     # dog verify trips
        self.t, self.result, self._last_refresh = 0.0, None, -99
        self.counts = {k: 0 for k in (
            'brain_calls', 'brain_unparsed', 'plan_applied', 'plan_rejected', 'decisions',
            'jev_calls', 'vlm_local_calls', 'verify_calls', 'inspects', 'dog_dispatches',
            'dog_requests', 'rejects', 'declares', 'recharges', 'safety_recharges',
            'battery_outs', 'frontier_fallbacks', 'service_errors')}
        self.latency = {'brain': [], 'jev': [], 'verify': [], 'vlm_local': []}
        self.explored = []
        self.out = Path(out_dir) if out_dir else None
        self._events = (self.out / 'events.jsonl').open('w') if self.out else None
        self.video = None
        if video and self.out:
            import imageio
            self.video = imageio.get_writer(str(self.out / 'episode.mp4'), fps=10, macro_block_size=8, quality=7)
        self.hud = {'brain': 'waiting for first plan', 'local': '', 'why': {}}

    # ------------------------------------------------------------ plumbing
    def log(self, event, **kw):
        if self._events:
            self._events.write(json.dumps({'t': round(self.t, 2), 'event': event, **kw}, default=_jsonable) + '\n')

    def schedule(self, release_t, kind, payload):
        self._seq += 1
        self.pending.append((release_t, self._seq, kind, payload))
        self.pending.sort(key=lambda p: (p[0], p[1]))

    def due(self):
        out = []
        while self.pending and self.pending[0][0] <= self.t + 1e-9:
            out.append(self.pending.pop(0))
        return out

    def charge(self, latency, on_vlm):
        """Release time for an answer computed now with this measured latency."""
        if self.clock == 'free':
            return self.t
        return self.server.schedule(self.t, latency) if on_vlm else self.t + latency

    def role(self, name):
        return 'dog' if name == DOG else 'drone'

    def body(self, name):
        return self.world.bodies[name]

    def active(self, name):
        b = self.body(name)
        return b.state not in ('down', 'docked')

    def set_task(self, name, task, why=''):
        old = self.tasks.get(name)
        if old and old['kind'] == 'verify':
            self.close_trip(old, why)
        # A new task starts its own arrival clock (a stale one triggered repositions 4 s into a
        # fresh trip) and, for the dog, its own collision memory.
        for tk_ in (old, task):
            if tk_ and tk_.get('target'):
                self.arrived_at.pop((name, tk_['target']), None)
        m = self.world.map
        if name == DOG and getattr(m, 'collision', None) is not None:
            m.collision[:] = False
            m._cache = None
        self.tasks[name] = task
        b = self.body(name)
        b.set_goal(task.get('goal') if task else None, self.t)
        b.face_target = None
        if task and task['kind'] == 'verify':
            self.counts['dog_dispatches'] += 1
            task['trip'] = {'track': task['target'], 'start_t': self.t, 'start_travel': b.travelled,
                            'source': task['source']}
            task['trip_origin'] = (b.x, b.z)
            if task['target'] in self.cand:
                self.cand[task['target']]['dog_attempts'] += 1
        self.log('task', robot=name, task=_view(task), previous=_view(old), why=why)

    def close_trip(self, task, why):
        tr = self.world.tracker.tracks.get(task['target'])
        trip = dict(task['trip'])
        trip.update({'end_t': self.t, 'walked_m': round(self.body(DOG).travelled - trip['start_travel'], 2),
                     'ended': why, 'candidate_real': bool(tr and self.world.track_is_real(tr)),
                     'declared': bool(self.result and self.result.get('track') == task['target'])})
        trip['empty_run'] = not (trip['declared'] and self.result and self.result.get('success'))
        # An empty run proper is a trip to something that was not the target;
        # failing to finish on a real one is a different failure.
        trip['wasted_on_false'] = trip['empty_run'] and not trip['candidate_real']
        self.trips.append(trip)
        self.log('dog_trip', **trip)

    # ------------------------------------------------------------ frontiers
    def refresh(self):
        for n in self.names:
            b = self.body(n)
            if b.state == 'down':
                continue
            self.fields[n] = self.world.map.distance_field(b.x, b.z, b.layer)
        self.registry.update(self.world, {n: f for n, f in self.fields.items() if self.body(n).state != 'down'})
        if self.sched == 'hybrid':
            # Time for each drone to reach the dog along real paths (or to meet it at a
            # rendezvous): the straight-line recall estimate was too optimistic when the
            # dog stood where drones cannot fly.
            dog = self.body(DOG)
            for n in DRONES:
                if self.body(n).state != 'airborne' or n not in self.fields:
                    self._home[n] = None
                    continue
                dock = self.dock_cell(n, dog)
                if dock is not None:
                    i, j = self.world.map.cell(*dock)
                    d = self.world.map.dist_at(self.fields[n], np.array([i]), np.array([j]))[0]
                    self._home[n] = float(d) / self.body(n).speed if np.isfinite(d) else None
                else:
                    self._home[n] = self.meet_time(n)
        if self.sched == 'hybrid' and (self.brain is not None or self.jev_ranker is not None):
            # Keep the ranking only over live exits; ask again when new exits appear.
            self.order = [f for f in self.order if f in self.registry.active]
            shown = sorted(self.registry.active.values(), key=lambda f: min(f['cost'].values()))[:board.MAX_FRONTIERS]
            stale = any(f['id'] not in self.order for f in shown) and self.t - self.order_t >= 6.0
            if self.pt:
                # Jev scores the shown exits whenever new ones appear; the brain keeps its own cadence for tasks
                if any(f['id'] not in self.fp for f in shown) and self.t - self.fp_t >= 6.0 and not self.rank_inflight:
                    self.fp_t = self.t
                    self.jev_rank(shown)
                if stale and self.brain is not None:
                    self.brain_wanted = True
            elif stale and self.jev_ranker is not None and not self.rank_inflight:
                self.jev_rank(shown)
            elif stale and self.brain is not None:
                self.brain_wanted = True

    def jev_rank(self, shown):
        """J3: Jev ranks the shown frontiers by where the target is likely; the
        order lands after Jev's measured latency (charged, not on the VLM queue)."""
        from jev_rank import rank_frontiers
        fr = [{'id': f['id'], 'width_m': round(f['size'] * self.world.map.res, 1),
               'landmarks_near': tb.nearby_labels(self.world, *f['centroid'])} for f in shown]
        t0 = time.perf_counter()
        try:
            order, p = rank_frontiers(self.jev_ranker, self.target, fr)
        except Exception as exc:
            if not is_service_error(exc):
                raise
            self.counts['service_errors'] += 1
            self.log('service_error', service='jev_rank', error=f'{type(exc).__name__}: {exc}')
            self.order_t = self.t                      # back off
            return
        latency = time.perf_counter() - t0
        self.counts['jev_calls'] += 1
        self.latency['jev'].append(latency)
        self.rank_inflight = True
        self.log('jev_rank', order=order, p={k: round(v, 3) for k, v in p.items()}, latency_s=round(latency, 3))
        self.schedule(self.charge(latency, False), 'jev_order', {'order': order, 'p': p})

    def dock_cell(self, name, dog):
        """Nearest air-reachable cell within docking range of the dog."""
        f = self.fields.get(name)
        if f is None:
            return None
        m = self.world.map
        i, j = m.cell(dog.x, dog.z)
        r = int(np.ceil(DOCK_RADIUS / m.res))
        ii, jj = np.mgrid[i - r:i + r + 1, j - r:j + r + 1]
        wx, wz = m.world(ii, jj)
        sep = np.hypot(wx - dog.x, wz - dog.z)
        dist = m.dist_at(f, ii, jj)
        ok = np.isfinite(dist) & (sep <= DOCK_RADIUS - 0.15)
        if not ok.any():
            return None
        k = np.argmin(np.where(ok, dist + sep, np.inf))
        return float(wx.ravel()[k]), float(wz.ravel()[k])

    def meet_time(self, name):
        pt = self.rendezvous(name, with_time=True)
        return pt[2] if pt else None

    def rendezvous(self, name, with_time=False):
        """Meeting point for a recharging drone and the dog: among cells both can
        reach on their own layers, the one minimising the later of the two arrival
        times."""
        fa, fg = self.fields.get(name), self.fields.get(DOG)
        if fa is None or fg is None or not self.active(DOG):
            return None
        m = self.world.map
        ok = np.isfinite(fg['dist'])
        if not ok.any():
            return None
        ii, jj = np.nonzero(ok)
        gd = fg['dist'][ok]
        ii, jj = ii + fg['i0'], jj + fg['j0']
        ad = m.dist_at(fa, ii, jj)
        t = np.maximum(ad / self.body(name).speed, gd / self.body(DOG).speed)
        if not np.isfinite(t).any():
            return None
        k = int(np.argmin(t))
        wx, wz = m.world(ii[k], jj[k])
        return (float(wx), float(wz), float(t[k])) if with_time else (float(wx), float(wz))

    def reposition(self, tk, tr, st, b):
        """The dog has stood at a candidate for REPOSITION_S with nothing left to do (in range, no
        own confirmation, no more looks): stand elsewhere, at least 60 deg round, and look again.
        Sitting there until the trip timed out cost 22 of 30 stalled trips (HM3D val run); four of
        those dogs were already on a success spot."""
        if self.t - st.get('repos_try_t', -math.inf) < 2.0 or st.get('repos_n', 0) >= MAX_REPOSITIONS:
            return False
        st['repos_try_t'] = self.t
        # Avoid the side it stood on; a goal it never reached may be the only side that works.
        g = self.standoff(DOG, tr, avoid=tk['goal']) or self.standoff(DOG, tr)
        if not g or math.hypot(g[0] - b.x, g[1] - b.z) < 0.5:
            return False
        tk['goal'] = st['last_goal'] = g
        st['repos_n'] = st.get('repos_n', 0) + 1
        b.set_goal(g, self.t)
        self.arrived_at.pop((DOG, tr.track_id), None)
        st['own_reset'] = sum(1 for c in tr.vlm if c.get('by') == DOG)
        self.log('reposition', robot=DOG, track=tr.track_id, goal=[round(v, 2) for v in g])
        return True

    def seen_here(self, name, tr):
        """The dog's own look named the target from (within VIEW_AT_M of) where it stands."""
        b = self.body(name)
        return name == DOG and any(c.get('by') == name and c['category'] == self.target and 'at' in c and
                                   math.hypot(c['at'][0] - b.x, c['at'][1] - b.z) <= VIEW_AT_M for c in tr.vlm)

    def reach(self, name, tr):
        return self.world.reach(name, tr, seen_here=self.seen_here(name, tr))

    def _should_look(self, n, tr, st, en_route=False):
        """The dog asks its own second opinion until it has a clear one (max 3):
        en route whenever it has a fresh detection within LOOK_NEAR_M, at the
        standoff always (a look-at crop if the detector has not fired). Looks
        taken before a reposition do not count."""
        own = [c for c in tr.vlm if c.get('by') == n][st.get('own_reset', 0):]
        if st['check_pending'] or st['status'] != 'open' or len(own) >= 3 or any(c['clear'] for c in own) or \
                self.t - st.get('look_t', -math.inf) <= LOOK_EVERY_S:
            return False
        if en_route:
            crop = tr.last_crop.get(n)
            loc = self.world.locate(tr)
            b = self.body(n)
            return bool(crop and crop[0] >= self.t - FRESH_VIEW_S and
                        math.hypot(loc[0] - b.x, loc[2] - b.z) <= LOOK_NEAR_M)
        # At its standoff it always looks: the map's sight line can miss the object (reach > 1 m),
        # and gating the look on it left dogs choosing "approach" in place until the trip timed out
        # without one look (an earlier HM3D val run: 10 trips, 0 m walked).
        return self.reach(n, tr) <= ag.DECLARE_RANGE or self.body(n).status in ('arrived', 'blocked')

    def _should_look_around(self, tr, st):
        own = [c for c in tr.vlm if c.get('by') == DOG][st.get('own_reset', 0):]
        crop = tr.last_crop.get(DOG)
        return (not st['check_pending'] and st['status'] == 'open' and st.get('scan_n', 0) < 2 and
                len(own) >= 2 and all(c.get('category') != self.world.episode['target'] for c in own[-2:]) and
                not (crop and crop[0] >= self.t - FRESH_VIEW_S))

    def usable(self, name):
        return {fid: f for fid, f in self.registry.active.items()
                if self.blacklist[name].get(fid, -1) < self.t and name in f['goal']}

    def frontier_task(self, name, fid, source):
        f = self.registry.active.get(fid)
        if f is None or name not in f['goal']:
            return None
        goal = f['goal'][name]
        if self.body(name).aerial and not ag.affordable(self.world, name, goal):
            return None
        return {'kind': 'explore', 'target': fid, 'goal': goal, 'source': source, 'since': self.t}

    # ------------------------------------------------------------ candidates
    def candidates_view(self):
        out = []
        dog = self.body(DOG)
        for tid, st in self.cand.items():
            tr = self.world.tracker.tracks[tid]
            s = tr.summary(self.t)
            out.append({'id': tid, 'label': tr.label, 'views': s['views'], 'max_score': s['max_score'],
                        'xz': s['xz'], 'status': st['status'], 'handled_by': st['handled_by'],
                        'dog_requested': st['dog_requested'],
                        'dog_distance_m': round(math.hypot(s['xz'][0] - dog.x, s['xz'][1] - dog.z), 1),
                        'second_opinions': [{'category': c['category'], 'clear': c['clear'],
                                             'description': (c['description'] or '')[:120]} for c in tr.vlm]})
        return out

    def save_crop(self, name, tr, crop, ans, asked):         # crop: (time taken, jpeg bytes)
        """SCOUT_CROPS sidecar: the verifier's input crop, its answer and the annotation under the detection box
        (None for a look-at crop, which has no box). Never read by a policy."""
        d = self.out / 'crops'
        d.mkdir(exist_ok=True)
        k = self.counts['verify_calls']
        (d / f'{k:03d}.jpg').write_bytes(crop[1])
        gt = getattr(tr, 'crop_gt', {}).get(name)
        rec = {'k': k, 'file': f'crops/{k:03d}.jpg', 't': round(self.t, 2), 'robot': name, 'track': tr.track_id,
               'target': self.target, 'detector_label': tr.label, 'track_was_real': self.world.track_is_real(tr),
               'crop_t': round(crop[0], 2), 'gt_top': gt[1] if gt and abs(gt[0] - crop[0]) < 1e-6 else None,
               'answer': {x: ans.get(x) for x in ('category', 'clear', 'description', 'reflection', 'raw')}, **asked}
        with open(d / 'crops.jsonl', 'a') as fh:
            fh.write(json.dumps(rec) + '\n')

    def standoff(self, name, tr, lo=0.35, hi=0.85, fallback=ag.DECLARE_RANGE, avoid=None):
        """A reachable cell to stand at. The default (the dog's verify) must be
        within declaring range of the surface or there is none: a 2.5 m
        fallback made an unreachable toilet look verifiable, and the scheduler
        sent the dog to stand 2.8 m from it nine times (study attempt 1)."""
        f = self.fields.get(name)
        if f is None:
            return None
        if name == DOG and getattr(tr, 'cells', None):
            return self.cell_standoff(tr, f, avoid)     # no point-based fallback: it has no sight line
        m = self.world.map
        p = self.world.locate(tr)
        i, j = m.cell(p[0], p[2])
        r = 25
        ii, jj = np.mgrid[i - r:i + r + 1, j - r:j + r + 1]
        dist = m.dist_at(f, ii, jj)
        wx, wz = m.world(ii, jj)
        sep = np.hypot(wx - p[0], wz - p[2])
        # Measured from the object's estimated near surface, not its centre:
        # a sofa's centre has no free cell within 0.85 m, and the fallback left
        # the dog 1.4 m from the surface, unable to declare (a development run).
        h = self.world.half_extent(tr)
        for a, b in ((lo + h, hi + h), (0.0, fallback + h)):
            ok = np.isfinite(dist) & (sep >= a) & (sep <= b)
            if avoid is not None:               # a standoff the robot got stuck on: try another side
                ok &= np.hypot(wx - avoid[0], wz - avoid[1]) > 0.6
            if ok.any():
                k = np.argmin(np.where(ok, dist + 2.0 * sep, np.inf))
                return float(wx.ravel()[k]), float(wz.ravel()[k])
        return None

    def scout_goal(self, tr):
        """No standoff yet: the floor round the candidate is unmapped (the dog waits at the start and the
        drones map little floor in a large building; zsNo4HB9uLZ:v2s20 dropped its only real target after
        20 s of no standoff). The reachable ground cell nearest the candidate, to walk to and map on the
        way; None if that is where the dog already stands."""
        f = self.fields.get(DOG)
        if f is None:
            return None
        m = self.world.map
        ii, jj = np.mgrid[0:m.n, 0:m.n]
        ok = m._derived()['ground'] & np.isfinite(m.dist_at(f, ii, jj))
        if not ok.any():
            return None
        p = self.world.locate(tr)
        wx, wz = m.world(ii, jj)
        k = np.argmin(np.where(ok, np.hypot(wx - p[0], wz - p[2]), np.inf))
        g = float(wx.ravel()[k]), float(wz.ravel()[k])
        b = self.body(DOG)
        return g if math.hypot(g[0] - b.x, g[1] - b.z) > 0.5 else None

    def cell_standoff(self, tr, field, avoid=None):
        """Where the dog stands for a candidate: a traversable cell 0.2-0.6 m from the object's own
        cells, with no occupied cell between it and the nearest of them, cheapest for the dog to
        walk to. `avoid` (a standoff that did not work) excludes bearings within 60 deg of it.
        A standoff picked by straight-line distance alone put the dog behind a bathroom wall."""
        from scipy import ndimage
        m = self.world.map
        der = m._derived()
        obj = np.asarray(list(tr.cells))
        r = int(np.ceil(STAND_MAX_M / m.res)) + 2
        i0, i1 = max(obj[:, 0].min() - r, 0), min(obj[:, 0].max() + r + 1, m.n)
        j0, j1 = max(obj[:, 1].min() - r, 0), min(obj[:, 1].max() + r + 1, m.n)
        mask = np.ones((i1 - i0, j1 - j0), bool)
        mask[obj[:, 0] - i0, obj[:, 1] - j0] = False
        d_obj, (ni, nj) = ndimage.distance_transform_edt(mask, return_indices=True)
        d_obj = d_obj * m.res
        ii, jj = np.mgrid[i0:i1, j0:j1]
        walk = m.dist_at(field, ii, jj)
        ok = der['ground'][i0:i1, j0:j1] & np.isfinite(walk) & (d_obj >= STAND_MIN_M) & (d_obj <= STAND_MAX_M)
        ci, cj = obj[:, 0].mean(), obj[:, 1].mean()
        if avoid is not None:
            ai, aj = m.cell(avoid[0], avoid[1])
            a0 = math.atan2(ai - ci, aj - cj)
            ang = np.abs((np.arctan2(ii - ci, jj - cj) - a0 + math.pi) % (2 * math.pi) - math.pi)
            ok &= ang > math.radians(60)
        from team_world import sight_line
        first = None
        for k in np.argsort(np.where(ok, walk, np.inf), axis=None)[:80]:
            a, b = np.unravel_index(k, ok.shape)
            if not ok[a, b]:
                break
            if first is None:
                first = (a, b)
            if sight_line(m, (ii[a, b], jj[a, b]), (ni[a, b] + i0, nj[a, b] + j0)):
                g = tuple(float(v) for v in m.world(ii[a, b], jj[a, b]))
                self._log_standoff(tr, obj, g, 'sight_line', walk[a, b])
                return g
        # No cell with a clear 2-D sight line (a plant on a table, a toilet against a wall: the map
        # marks everything round them): take the cheapest cell in the band anyway. A wrong-side
        # cell cannot cause a declaration -- reach() counts only object cells in sight -- the dog
        # just repositions; returning None left the dog idle all mission (retest BAbdmeyTvMZ:25).
        g = None if first is None else tuple(float(v) for v in m.world(ii[first], jj[first]))
        self._log_standoff(tr, obj, g, 'fallback', None if first is None else walk[first])
        return g

    def _log_standoff(self, tr, obj, g, how, walk):
        m = self.world.map
        w = np.array([m.world(i, j) for i, j in obj])
        self.log('standoff', track=tr.track_id, how=how, goal=None if g is None else [round(v, 2) for v in g],
                 walk_m=None if walk is None else round(float(walk), 2), cells=len(obj),
                 cells_min=[round(float(v), 2) for v in w.min(0)], cells_max=[round(float(v), 2) for v in w.max(0)])

    def ctx(self, name, tr):
        b, dog = self.body(name), self.body(DOG)
        p = self.world.locate(tr)
        d = math.hypot(p[0] - b.x, p[2] - b.z)
        st = self.cand[tr.track_id]
        crop = tr.last_crop.get(name)
        fresh = bool(crop and crop[0] >= self.t - FRESH_VIEW_S)
        mates = []
        for n in self.names:
            if n == name:
                continue
            o = self.body(n)
            mates.append({'id': n, 'type': self.role(n), 'task': board.task_text(self.tasks.get(n)),
                          'distance_to_candidate_m': round(math.hypot(p[0] - o.x, p[2] - o.z), 1)})
        return {'target': self.target, 'distance_m': d, 'path_m': d, 'fresh_view': fresh,
                'reach_m': self.reach(name, tr),
                'handled_by': st['handled_by'], 'check_pending': st['check_pending'],
                'dog_requested': st['dog_requested'],
                'near_labels': tb.nearby_labels(self.world, p[0], p[2], exclude=tr),
                'task_desc': board.task_text(self.tasks.get(name)), 'teammates': mates,
                'time_left_s': self.mission_s - self.t, 'battery_s': b.battery if b.aerial else 0.0,
                'dog_distance_m': math.hypot(p[0] - dog.x, p[2] - dog.z),
                'can_declare': self.world.can_declare(name), 'lessons': self.lessons,
                'inspected': bool(st.get('inspected_by', {}).get(name)),
                'own_confirmation': (self.sched != 'hybrid' or
                                     # names the target, clear or not: on the dog's own views (development data)
                                     # clear-only kept 9 real / 0 false, any clarity 19 real / 3 false
                                     # ...and from where it stands now: a look from its previous spot let
                                     # it declare from one that cannot see the vase (CrMo8WxCyVb:12)
                                     self.seen_here(name, tr))}

    def decide(self, name, tr, reason):
        st = self.cand[tr.track_id]
        if st['status'] != 'open' or self.result or not self.active(name):
            return
        # A reflection: drop the candidate and turn the robot around, where the real object is. Only when the
        # answer did not name the target: a TV's glossy screen "shows a reflection" and the flag alone rejected
        # real TVs team-wide (HM3D val: 21 episodes, e.g. p53SfW6mjZe:19).
        last = tr.vlm[-1] if reason == 'second_opinion_returned' and tr.vlm else None
        # Like every rejection, only under the two-key rule (P below the reject gate and contrary evidence):
        # this path skipped it and rejected 26 real targets team-wide (HM3D val rerun). Otherwise the
        # answer goes to the judge like any other.
        if last and last.get('reflection') and last.get('category') != self.target and \
                (st.get('p') if st.get('p') is not None else 1.0) < ag.REJECT_GATE and ag.reject_backed(tr, self.target):
            self.act(name, tr.track_id, 'reject')
            tk = self.tasks.get(name)
            if tk and tk.get('target') == tr.track_id:
                self.set_task(name, None, why='reflection: turn around')
            self.spin_left[name] = math.pi / (3.0 if self.body(name).aerial else 2.0)   # Body.spin rad/s
            self.log('reflection_turn', robot=name, track=tr.track_id)
            return
        key = (name, tr.track_id)
        if reason == 'new_view' and self.t - self.last_decision.get(key, -math.inf) < DECISION_COOLDOWN_S:
            return
        role = self.role(name)
        dec = self.dec[role]
        if dec.kind != 'rule' and self.inflight[name]:
            self.deferred[name] = (tr.track_id, reason)     # re-run once the call in flight lands
            return
        if dec.kind == 'rule' and max(tr.scores) < ag.RULE_APPROACH_SCORE and not tr.vlm:
            return
        self.last_decision[key] = self.t
        ctx = self.ctx(name, tr)
        try:
            action, latency, rec = dec.candidate(role, name, tr, ctx, self.t)
        except Exception as exc:
            if not is_service_error(exc):
                raise                    # a bug (bad value, wrong type), not the service: stop the run
            self.counts['service_errors'] += 1
            self.log('service_error', service=dec.kind, robot=name, error=f'{type(exc).__name__}: {exc}')
            return
        self.counts['decisions'] += 1
        if dec.kind == 'jev':
            self.counts['jev_calls'] += 1
            self.latency['jev'].append(latency)
        elif dec.kind == 'vlm':
            self.counts['vlm_local_calls'] += 1
            self.latency['vlm_local'].append(latency)
        if dec.kind != 'rule':            # no model call to wait for
            self.inflight[name] = True
        self.log('candidate_decision', robot=name, role=role, track=tr.track_id, reason=reason,
                 action=action, latency_s=round(latency, 3), record=rec,
                 context={k: v for k, v in ctx.items() if k != 'teammates'})
        self.hud['local'] = f"{name} on {tr.track_id}: {action}" + (
            f"  P={rec['p_target']:.2f}" if 'p_target' in rec else '')
        self.schedule(self.charge(latency, dec.kind == 'vlm'), 'candidate_action',
                      {'robot': name, 'track': tr.track_id, 'action': action, 'p': rec.get('p_target')})

    def act(self, name, tid, action):
        self._act(name, tid, action)
        # Dispatch gate: any candidate whose calibrated P clears it sends the dog, not only an explicit drone
        # request -- when the decision is released (its latency is charged); the method's scheduler only.
        st = self.cand.get(tid)
        if self.sched != 'hybrid' or self.allocation != 'brain' or not st or self.result or \
                st['status'] != 'open' or st['dog_requested'] or (st.get('p') or 0.0) < ag.DOG_REQUEST_GATE:
            return
        st['dog_requested'] = True
        self.counts['dog_requests'] += 1
        self.log('dog_requested', track=tid, by=f"auto P={st['p']:.2f}")
        self.dispatch_requested()

    def _act(self, name, tid, action):
        tr = self.world.tracker.tracks[tid]
        st = self.cand[tid]
        if st['status'] != 'open' or not self.active(name):
            return
        b = self.body(name)
        if action == 'reject':
            st['status'] = 'rejected'
            self.counts['rejects'] += 1
            for n, tk in self.tasks.items():
                if tk and tk.get('target') == tid:
                    self.set_task(n, None, why=f'{tid} rejected')
            st['handled_by'] = None
            self.log('candidate_rejected', track=tid, by=name)
        elif action == 'continue_task':
            pass                # keep doing the current task, as the option says
        elif action == 'inspect':
            if st['handled_by'] not in (None, name) or not b.aerial:
                return
            tk = self.tasks.get(name)
            if tk and tk['kind'] == 'inspect' and tk.get('target') == tid:
                return          # already on it; restarting reset its time limit (a development run: 100 s hover)
            if st.get('inspected_by', {}).get(name):
                return          # this drone already had its close look
            goal = self.standoff(name, tr, 0.9, 1.6, 3.0)
            if goal is None or not ag.affordable(self.world, name, goal):
                return
            st['handled_by'] = name
            self.counts['inspects'] += 1
            self.set_task(name, {'kind': 'inspect', 'target': tid, 'goal': goal, 'source': 'local',
                                 'since': self.t}, why='inspect candidate')
        elif action == 'request_dog':
            if not st['dog_requested']:
                st['dog_requested'] = True
                self.counts['dog_requests'] += 1
                self.log('dog_requested', track=tid, by=name)
                if self.allocation != 'geometric' and self.sched == 'hybrid':
                    # The request already passed the local calibrated gate: code sends the
                    # dog; the scheduler is told on its next call.
                    self.dispatch_requested()
                    self.brain_wanted = True
                elif self.allocation == 'brain':
                    self.brain_wanted = True
                else:
                    self.geometric_dispatch()
        elif action == 'approach':
            if name != DOG:
                return self.act(name, tid, 'inspect')
            tk = self.tasks.get(name)
            if self.sched == 'hybrid' and not (tk and tk['kind'] == 'verify') and \
                    st['dog_attempts'] < MAX_DOG_ATTEMPTS:
                # The dog saw it itself and its (gated) decision is to go: it takes the
                # verify task. (validation buildings: the dog judged a real toilet at 2.5 m
                # with P = 0.54, chose "approach" and nothing happened.)
                goal = self.standoff(name, tr)
                if goal:
                    self.set_task(name, {'kind': 'verify', 'target': tid, 'goal': goal,
                                         'source': 'self', 'since': self.t}, why='dog approaches what it saw')
                return
            if not (tk and tk['kind'] == 'verify' and tk['target'] == tid):
                return          # already verifying another candidate
            if b.status == 'moving' and tk.get('goal') is not None:
                return          # already walking to a standoff (possibly a reposition): keep it
            ov = getattr(tr, 'own_view', None)
            if ov and self.t - ov['t'] <= FRESH_VIEW_S:
                # Its own detector has the object right now: walk at that point. The cell-based standoff can be where
                # it already stands, 1.25 m away, and it chose "approach" 11 times in 55 s (mL8ThkuaVTM:0).
                goal = self.final_goal(name, tr)
                if goal:
                    tk['goal'] = goal
                    b.set_goal(None, self.t); b.set_goal(goal, self.t)
                    return
            stuck = b.status == 'blocked' and tk.get('goal') is not None
            goal = (self.standoff(name, tr, avoid=tk['goal']) if stuck else None) or self.standoff(name, tr)
            if goal and b.status == 'arrived' and math.hypot(goal[0] - b.x, goal[1] - b.z) < 0.3:
                # "Closer" resolves to where it already stands: this spot does not work, stand elsewhere.
                self.reposition(tk, tr, st, b)
            elif goal:
                tk['goal'] = goal
                b.set_goal(goal, self.t)
        elif action == 'second_opinion':
            crop = tr.last_crop.get(name)
            if (not crop or crop[0] < self.t - FRESH_VIEW_S) and name == DOG and self.sched == 'hybrid' and \
                    (self.reach(name, tr) <= ag.DECLARE_RANGE or self.body(name).status in ('arrived', 'blocked')):
                look = self.world.look_crop(name, tr)          # the dog looks where the candidate should be
                crop = (self.t, look) if look else None
                if look and self.out:
                    self.counts['looks'] = self.counts.get('looks', 0) + 1
                    if self.counts['looks'] <= 30:
                        (self.out / f"look_{self.counts['looks']:02d}_{tid}.jpg").write_bytes(look)
            if not crop or crop[0] < self.t - FRESH_VIEW_S or st['check_pending']:
                return
            st['check_pending'] = True
            try:
                ans = verify(self.server, crop[1], self.vocabulary)
                asked = {'prompt': ans.pop('prompt', None), 'vocabulary': ans.pop('vocabulary', None)}   # sidecar only
            except Exception as exc:
                st['check_pending'] = False
                if not is_service_error(exc):
                    raise
                self.counts['service_errors'] += 1
                self.log('service_error', service='verify', error=f'{type(exc).__name__}: {exc}')
                return
            self.counts['verify_calls'] += 1
            self.latency['verify'].append(ans['latency_s'])
            release = self.charge(ans['latency_s'], True)
            ob = self.body(name)
            ans.update({'by': name, 't': round(release, 2), 'issued': round(self.t, 2),
                        'at': [round(ob.x, 2), round(ob.z, 2)]})     # where the view was taken
            self.log('verify_request', robot=name, track=tid, release_t=round(release, 2),
                     latency_s=round(ans['latency_s'], 3))
            if os.environ.get('SCOUT_CROPS') and self.out:     # evaluation sidecar: training data for the verifier
                self.save_crop(name, tr, crop, ans, asked)
            self.schedule(release, 'verify_reply', {'robot': name, 'track': tid, 'answer': ans})
        elif action == 'declare_found':
            if not self.world.can_declare(name) or self.reach(name, tr) > ag.DECLARE_RANGE or st.get('final_approach'):
                return
            if name == DOG and self.sched == 'hybrid':
                goal = self.final_goal(name, tr)
                if goal is not None:
                    # Confirmed and in range: walk up to it before declaring, as an ObjectNav agent stops by the
                    # object -- declaring from 0.6-1.0 m missed the view points by 0.13-0.32 m (38 episodes).
                    st['final_approach'] = {'t0': self.t, 'from': [round(b.x, 2), round(b.z, 2)], 'goal': [round(v, 2) for v in goal],
                                            'limit_s': round(FINAL_APPROACH_S + 2.0 * math.hypot(goal[0] - b.x, goal[1] - b.z), 1)}
                    tk = self.tasks.get(name)
                    if tk: tk['goal'] = goal
                    # Cleared first: set_goal keeps a goal within 0.3 m of the old one and its stale 'arrived'
                    # status, which declared on the spot (mL8ThkuaVTM:0, goal 0.19 m away, dtg 0.165 m).
                    b.set_goal(None, self.t); b.set_goal(goal, self.t)
                    self.log('final_approach', robot=name, track=tid, **st['final_approach'])
                    return
            self.declare(name, tid, tr)

    def final_goal(self, name, tr):
        """A traversable cell FINAL_STANDOFF_M short of the object on the robot's line to its
        aim point (what its look confirmed); None if already within FINAL_APPROACH_M. Not the nearest object cell:
        on a merged 178-cell track it sat 0.8 m from the sofa's view points (a8BtkwhxdRV:10)."""
        b = self.body(name); m = self.world.map
        p = self.world.aim_point(name, tr); tx, tz = float(p[0]), float(p[2])
        d = math.hypot(tx - b.x, tz - b.z)
        if d <= FINAL_APPROACH_M:
            return None
        px, pz = tx - (tx - b.x) / d * FINAL_STANDOFF_M, tz - (tz - b.z) / d * FINAL_STANDOFF_M
        ground = m._derived()['ground']; gi, gj = m.cell(px, pz); r = int(0.5 / m.res)
        i0, j0 = max(gi - r, 0), max(gj - r, 0)
        ii, jj = np.nonzero(ground[i0:gi + r + 1, j0:gj + r + 1])
        if ii.size == 0:
            return None
        k = int(np.argmin((ii + i0 - gi) ** 2 + (jj + j0 - gj) ** 2))
        g = tuple(float(v) for v in m.world(ii[k] + i0, jj[k] + j0))
        return g if math.hypot(g[0] - b.x, g[1] - b.z) > 0.1 else None

    def declare(self, name, tid, tr):
        st = self.cand[tid]
        if self.result:
            return
        self.counts['declares'] += 1
        score = self.world.score_declaration(name)
        aim = [round(float(v), 2) for v in self.world.aim_point(name, tr)[[0, 2]]]
        self.result = {'outcome': 'success' if score['success'] else 'false_declaration',
                       'declared_by': name, 'track': tid, 't': round(self.t, 2),
                       'track_was_real': self.world.track_is_real(tr), **score, 'aim_xz': aim,
                       **({'final_approach': st['final_approach']} if st.get('final_approach') else {})}
        self.log('declare', **self.result)
        tk = self.tasks.get(name)
        if tk and tk['kind'] == 'verify':
            self.close_trip(tk, 'declared')
            self.tasks[name] = None

    # ------------------------------------------------------------ scheduling
    def dispatch_requested(self):
        """Hybrid: send the dog to a requested open candidate (a request that passed its gate):
        the most probable first, then the nearest; one without a reachable standoff is skipped
        (it used to block every other request) and dropped after NO_STANDOFF_S. Never while the
        dog is verifying or at a rendezvous."""
        cur = self.tasks.get(DOG)
        if cur and cur['kind'] in ('verify', 'meet') or not self.active(DOG):
            return
        dog = self.body(DOG)
        order = []
        for tid, st in self.cand.items():
            if st['status'] != 'open' or not st['dog_requested'] or st['dog_attempts'] >= MAX_DOG_ATTEMPTS:
                continue
            loc = self.world.locate(self.world.tracker.tracks[tid])
            order.append((-st.get('p', 0.0), math.hypot(loc[0] - dog.x, loc[2] - dog.z), tid))
        for _, _, tid in sorted(order):
            st, tr = self.cand[tid], self.world.tracker.tracks[tid]
            goal = (self.standoff(DOG, tr, avoid=st['last_goal']) if st.get('last_goal') else None) or \
                self.standoff(DOG, tr)
            if goal:
                st.pop('no_standoff_t', None)
                st['last_goal'] = goal
                self.set_task(DOG, {'kind': 'verify', 'target': tid, 'goal': goal,
                                    'source': 'requested', 'since': self.t}, why='dog requested by a drone (gated)')
                return
            scout = self.scout_goal(tr)
            if scout:
                self.set_task(DOG, {'kind': 'verify', 'target': tid, 'goal': scout, 'scout': True, 'scout_t': self.t,
                                    'source': 'requested', 'since': self.t}, why='dog walks toward it: no standoff yet')
                return
            st.setdefault('no_standoff_t', self.t)
            if self.t - st['no_standoff_t'] > NO_STANDOFF_S:
                st['dog_attempts'] = MAX_DOG_ATTEMPTS
                self.log('no_standoff', track=tid, robot=DOG)

    def geometric_dispatch(self):
        """Geometric baseline: send the dog to any sufficiently scored candidate."""
        if self.tasks.get(DOG) and self.tasks[DOG]['kind'] == 'verify':
            return
        dog = self.body(DOG)
        best, best_d = None, math.inf
        for tid, st in self.cand.items():
            if st['status'] != 'open':
                continue
            tr = self.world.tracker.tracks[tid]
            if max(tr.scores) < GEOMETRIC_DISPATCH_SCORE and not st['dog_requested']:
                continue
            if st['dog_attempts'] >= 1:
                continue        # one trip per candidate (a development run re-sent the dog four times)
            d = math.hypot(tr.position[0] - dog.x, tr.position[2] - dog.z)
            if d < best_d:
                best, best_d = tr, d
        if best is None:
            return
        goal = self.standoff(DOG, best)
        if goal:
            self.set_task(DOG, {'kind': 'verify', 'target': best.track_id, 'goal': goal,
                                'source': 'geometric', 'since': self.t}, why='geometric dispatch')

    def assign_if_needed(self, name):
        if self.tasks.get(name) or self.spin_left[name] > 0 or not self.active(name):
            return
        if self.sched == 'hybrid':
            if name == DOG:
                if self.allocation == 'geometric':
                    self.geometric_dispatch()
                else:
                    self.dispatch_requested()
                if self.tasks.get(DOG):
                    return
            self._idle.append(name)          # allocated jointly after this pass
            return
        if self.allocation == 'geometric':
            if name == DOG:
                self.geometric_dispatch()
                if self.tasks.get(DOG):
                    return
            f = ag.pick_frontier(name, self.usable(name), self.tasks)
            if f:
                task = self.frontier_task(name, f['id'], 'geometric')
                if task:
                    self.set_task(name, task, why='geometric')
            return
        self.brain_wanted = True
        if self.frontier_inflight[name]:
            return
        fronts = self.usable(name)
        if not fronts:
            return
        dec = self.dec[self.role(name)]
        try:
            fid, latency, rec = dec.frontier(name, fronts, self.tasks, self.world, self.target, self.t)
        except Exception as exc:
            if not is_service_error(exc):
                raise
            self.counts['service_errors'] += 1
            self.log('service_error', service='frontier', error=f'{type(exc).__name__}: {exc}')
            f = ag.pick_frontier(name, fronts, self.tasks)
            fid, latency, rec = (f['id'] if f else None), 0.0, {'policy': 'geometric_after_error'}
        if dec.kind == 'jev' and latency > 0:
            self.counts['jev_calls'] += 1
            self.latency['jev'].append(latency)
        if dec.kind == 'vlm' and latency > 0:
            self.counts['vlm_local_calls'] += 1
            self.latency['vlm_local'].append(latency)
        self.frontier_inflight[name] = True
        self.log('local_frontier', robot=name, choice=fid, latency_s=round(latency, 3), record=rec)
        self.schedule(self.charge(latency, dec.kind == 'vlm'), 'frontier_choice', {'robot': name, 'fid': fid})

    def allocate(self, names, reassign=False):
        """Jointly send robots to frontiers (Hungarian). With a brain, its
        search_order adds RANK_S per rank step; without one, pure geometry. With
        DOG_EXPLORES off, the dog waits and moves only for a gated candidate."""
        from scipy.optimize import linear_sum_assignment
        fr = list(self.registry.active.values())
        names = [n for n in names if self.active(n) and (DOG_EXPLORES or n != DOG)]
        if not fr or not names:
            return
        rank = {fid: i for i, fid in enumerate(self.order)}
        unranked = len(self.order)
        taken = {tk['target'] for n, tk in self.tasks.items()
                 if n not in names and tk and tk['kind'] == 'explore'}
        big = 1e6
        C = np.full((len(names), len(fr)), big)
        res = self.world.map.res
        pt = getattr(self, 'pt', False)
        prior = float(np.median(list(self.fp.values()))) if pt and self.fp else 0.5   # an exit Jev has not scored yet
        for i, n in enumerate(names):
            b, usable = self.body(n), self.usable(n)
            for j, f in enumerate(fr):
                if f['id'] not in usable:
                    continue
                if b.aerial and not ag.affordable(self.world, n, f['goal'][n]):
                    continue
                travel = (f['cost'][n] - SIZE_BONUS_M * min(f['size'] * res, 4.0)) / b.speed
                if pt:
                    C[i, j] = max(travel, MIN_TRAVEL_S) / max(self.fp.get(f['id'], prior), P_FLOOR)
                else:
                    C[i, j] = travel + RANK_S * rank.get(f['id'], unranked)
        # A frontier another explorer already holds is out of the assignment while a free one is usable: as a cost
        # (DUP_S) it was cheaper than two rank steps, and a lone idle drone joined the top-ranked frontier instead of
        # taking the third (MP3D 300 s: >= 2 drones on one frontier 42% of the time with a search order, 14% without).
        free = np.array([f['id'] not in taken for f in fr])
        rows, cols = linear_sum_assignment(np.where(free, C, big))
        chosen = {names[i]: fr[j] for i, j in zip(rows, cols) if C[i, j] < big and free[j]}
        for i, n in enumerate(names):            # more robots than free frontiers: share, with the penalty
            if n not in chosen and np.min(C[i]) < big:
                j = int(np.argmin(C[i] + DUP_S * ~free))
                chosen[n] = fr[j]
        for n, f in chosen.items():
            cur = self.tasks.get(n)
            if cur and cur['kind'] == 'explore' and cur.get('target') == f['id']:
                continue
            task = self.frontier_task(n, f['id'], 'brain_rank' if self.order else 'geometric')
            if task:
                self.set_task(n, task, why='allocated (search order)' if self.order else 'allocated')

    def reassign_explorers(self):
        """After a new search order: re-run the assignment over idle robots and
        explorers whose task is older than the hysteresis."""
        cand = [n for n in self.names if self.active(n) and
                (self.tasks.get(n) is None or
                 (self.tasks[n]['kind'] == 'explore' and self.t - self.tasks[n]['since'] >= EXPLORE_HYSTERESIS_S))]
        self.allocate(cand)

    def task_menu(self):
        """Per-robot menu of tasks that are feasible right now (SayCan-style
        affordance grounding): reachable through seen space, affordable on the
        drone's battery, allowed for the robot's role. The scheduler picks from
        its robot's menu and cannot name anything else. Recharge appears only
        below half battery; hold only when nothing else is feasible."""
        menu = {}
        for n in self.names:
            b = self.body(n)
            if b.state == 'down':
                continue
            if b.state == 'docked':
                if self.sched != 'hybrid':
                    menu[n] = [{'task': 'hold', 'target': None, 'note': 'docked, swapping battery'}]
                continue
            opts = []
            hybrid = self.sched == 'hybrid' and self.allocation != 'direct'
            for fid, f in ([] if hybrid else sorted(self.usable(n).items(), key=lambda kv: kv[1]['cost'][n])[:5]):
                goal = f['goal'][n]
                if b.aerial and not ag.affordable(self.world, n, goal):
                    continue
                cx, cz = f['centroid']
                opts.append({'task': 'explore', 'target': fid, 'distance_m': round(f['cost'][n], 1),
                             'objects_near': tb.nearby_labels(self.world, cx, cz)[:4]})
            open_ = sorted((tid for tid, st in self.cand.items() if st['status'] == 'open'),
                           key=lambda k: -max(self.world.tracker.tracks[k].scores))[:3]
            for tid in open_:
                st = self.cand[tid]
                tr = self.world.tracker.tracks[tid]
                loc = self.world.locate(tr)
                d = math.hypot(loc[0] - b.x, loc[2] - b.z)
                if b.aerial and st['handled_by'] in (None, n) and d < 15 and \
                        not st.get('inspected_by', {}).get(n) and \
                        ag.affordable(self.world, n, (float(loc[0]), float(loc[2]))):
                    opts.append({'task': 'inspect', 'target': tid, 'distance_m': round(d, 1)})
                backed = (st['dog_requested'] or d <= VERIFY_NEAR_M or
                          any(c['clear'] and c['category'] == self.target for c in tr.vlm))
                if n == DOG and st['dog_attempts'] < MAX_DOG_ATTEMPTS and (backed or not hybrid) and \
                        self.standoff(DOG, tr) is not None:
                    opts.append({'task': 'verify', 'target': tid, 'distance_m': round(d, 1)})
            if b.aerial and b.battery <= RECHARGE_BELOW_FRACTION * DRONE_ENDURANCE_S:
                opts.append({'task': 'recharge', 'target': None,
                             'note': f'{b.battery:.0f} s of flight left, {ag.flight_home_s(self.world, n):.0f} s to the dog'})
            if not opts:
                if hybrid:
                    continue                 # free: the allocator keeps it searching
                opts.append({'task': 'hold', 'target': None, 'note': 'nothing else is feasible'})
            menu[n] = opts
        return menu

    def maybe_brain(self):
        if self.brain is None or self.brain_inflight or self.t < BRAIN_FIRST_S:
            return
        if self.t - self.brain_last_reply < BRAIN_MIN_INTERVAL_S:
            return
        period = HYBRID_BRAIN_PERIOD_S if self.sched == 'hybrid' else BRAIN_PERIOD_S
        if not (self.brain_wanted or self.t - self.brain_last_reply >= period):
            return
        cands = self.candidates_view()
        snap = board.snapshot(self.world, self.registry, self.tasks, self.t, self.target, self.mission_s,
                              cands, self.counts['brain_calls'] + 1, self.plan_note)
        if self.lessons:
            snap['team_experience'] = self.lessons
        self._menu = self.task_menu()
        snap['options'] = {n: [' '.join(x for x in (o['task'], o['target'] or '',
                                                     f"({o['distance_m']} m)" if 'distance_m' in o else
                                                     f"({o['note']})") if x)
                               for o in opts] for n, opts in self._menu.items()}
        board.fit(snap, BOARD_CHAR_BUDGET)
        img = board.render(self.world, self.registry, self.tasks, cands)
        try:
            reply = self.brain.plan(snap, tb.png_bytes(img), self.target)
        except Exception as exc:
            if not is_service_error(exc):
                raise
            self.counts['service_errors'] += 1
            self.log('service_error', service='brain', error=f'{type(exc).__name__}: {exc}')
            # Back off: a failing call retried every tick produced 537 errors in one development run.
            self.brain_last_reply, self.brain_wanted = self.t, False
            return
        self.counts['brain_calls'] += 1
        self.latency['brain'].append(reply['latency_s'])
        release = self.charge(reply['latency_s'], True)
        self.brain_inflight, self.brain_wanted = True, False
        self.log('brain_request', board=snap, release_t=round(release, 2),
                 latency_s=round(reply['latency_s'], 3), usage=reply['usage'])
        if self.out and self.counts['brain_calls'] <= 60:
            img.save(self.out / f"board_{self.counts['brain_calls']:03d}.png")
        self.schedule(release, 'brain_reply', {'reply': reply})
        self.hud['brain'] = f'thinking (issued t={self.t:.1f}s)'

    def apply_plan(self, reply):
        self.brain_inflight, self.brain_last_reply = False, self.t
        plan = reply['assignments']
        if plan is None:
            self.counts['brain_unparsed'] += 1
            self.log('brain_reply', ok=False, raw=reply['raw'])
            self.hud['brain'] = 'reply unparseable, keeping tasks'
            return
        if self.sched == 'hybrid':
            self.order, self.order_t = [f for f in reply.get('order', []) if f in self.registry.active or
                                        self.registry.successor(f)], self.t
            # ids not on the board: the 8B brain copied the prompt's example (["F3", "F1"], T4) in
            # 21-49% of replies before the example became placeholders; count what is still dropped
            dropped = [f for f in reply.get('order', []) if f not in self.order]
            if dropped or '<' in reply['raw']:
                self.counts['brain_order_dropped'] = self.counts.get('brain_order_dropped', 0) + 1
                self.log('brain_order_dropped', dropped=dropped, placeholder='<' in reply['raw'])
            self.order = list(dict.fromkeys(self.registry.successor(f) or f for f in self.order))
        applied, rejected, used = [], [], set()
        for a in plan:
            n, kind, tgt = a['robot'], a['task'], a['target']
            if n in self.names and self.body(n).state == 'docked':
                continue                     # swapping batteries; nothing to schedule
            cur = self.tasks.get(n)
            if self.sched == 'hybrid' and cur and cur['kind'] in ('verify', 'inspect', 'recharge') and \
                    not (cur['kind'] == kind and cur.get('target') == tgt):
                rejected.append({**a, 'reason': 'busy: tasks under way are not interrupted'})
                continue
            reason = self.plan_check(n, kind, tgt, used)
            if reason:
                rejected.append({**a, 'reason': reason})
                continue
            task = self.plan_task(n, kind, tgt)
            if task is None:
                rejected.append({**a, 'reason': 'no feasible goal (unreachable or battery)'})
                continue
            cur = self.tasks.get(n)
            if cur and cur['kind'] == task['kind'] and cur.get('target') == task.get('target'):
                cur['source'] = 'brain'
            else:
                self.set_task(n, task, why=a['why'])
            if tgt:
                used.add(task.get('target'))
            self.hud['why'][n] = f"{kind}({task.get('target') or ''}): {a['why']}"
            applied.append({**a, 'target': task.get('target')})
        self.counts['plan_applied'] += len(applied)
        self.counts['plan_rejected'] += len(rejected)
        if self.sched == 'hybrid' and self.allocation == 'brain':
            self.reassign_explorers()
        self.plan_note = {'at_s': round(self.t, 1),
                          'assignments': [f"{a['robot']}:{a['task']}({a['target'] or ''})" for a in applied]}
        self.log('brain_reply', ok=True, raw=reply['raw'], applied=applied, rejected=rejected,
                 order=self.order if self.sched == 'hybrid' else None)
        self.hud['brain'] = f"plan at t={self.t:.1f}s: " + ', '.join(
            f"{a['robot']}:{a['task']}" for a in applied)

    def plan_check(self, n, kind, tgt, used):
        if n not in self.tasks:
            return 'unknown robot'
        menu = getattr(self, '_menu', {}).get(n)
        if menu is not None and not any(o['task'] == kind and (o['target'] or None) == (tgt or None) for o in menu):
            return 'not on this robot\'s menu'
        b = self.body(n)
        if b.state == 'down':
            return 'robot is down'
        if b.state == 'docked' and kind != 'recharge':
            return 'robot is docked'
        if kind in ('inspect', 'recharge') and not b.aerial:
            return f'{kind} is for drones'
        # A drone with most of its battery left gains nothing from a swap, and a
        # brain that keeps saying "recharge" otherwise parks it on the dog: in a
        # development run one drone docked eight times in a row at full charge.
        if kind == 'recharge' and b.battery > RECHARGE_BELOW_FRACTION * DRONE_ENDURANCE_S:
            return 'battery not low enough to recharge'
        if kind == 'verify' and n != DOG:
            return 'verify is for the dog'
        cur = self.tasks.get(n)
        if cur and cur['kind'] == 'recharge' and cur['source'] == 'safety' and kind != 'recharge':
            return 'battery safety recall in progress'
        if kind == 'explore' and tgt in used and len(self.registry.active) >= len(self.names):
            return 'duplicate frontier'
        if kind == 'inspect' and tgt in used:
            return 'candidate already given to another drone in this plan'
        return None

    def plan_task(self, n, kind, tgt):
        if kind == 'hold':
            return {'kind': 'hold', 'target': None, 'goal': None, 'source': 'brain', 'since': self.t}
        if kind == 'recharge':
            d = self.body(DOG)
            return {'kind': 'recharge', 'target': None, 'goal': (d.x, d.z), 'source': 'brain', 'since': self.t}
        if kind == 'explore':
            live = self.registry.successor(tgt) if tgt else None
            return self.frontier_task(n, live, 'brain') if live else None
        tr = self.world.tracker.tracks.get(tgt) if tgt else None
        if tr is None or tgt not in self.cand or self.cand[tgt]['status'] != 'open':
            return None
        if kind == 'inspect':
            goal = self.standoff(n, tr, 0.9, 1.6, 3.0)
            if goal is None or not ag.affordable(self.world, n, goal):
                return None
            self.cand[tgt]['handled_by'] = n
            self.counts['inspects'] += 1
            return {'kind': 'inspect', 'target': tgt, 'goal': goal, 'source': 'brain', 'since': self.t}
        goal = self.standoff(n, tr)
        if goal is None:
            return None
        return {'kind': 'verify', 'target': tgt, 'goal': goal, 'source': 'brain', 'since': self.t}

    # ------------------------------------------------------------ main loop
    def run(self):
        wall0 = time.perf_counter()
        steps = int(round(self.mission_s / CONTROL_DT))
        dog = self.body(DOG)
        for step in range(steps + 1):
            self.t = step * CONTROL_DT
            # Batteries and docking.
            for n in DRONES:
                b = self.body(n)
                ev = b.tick_battery(dog)
                if ev == 'battery_out':
                    self.counts['battery_outs'] += 1
                    self.tasks[n] = None
                    for st_ in self.cand.values():
                        if st_['handled_by'] == n:
                            st_['handled_by'] = None
                    self.log('battery_out', robot=n)
                elif ev == 'launched':
                    self.log('launched', robot=n)
                    self.spin_left[n] = 0.0
                tk = self.tasks.get(n)
                # The final landing on the dog's back is a precision-landing step we do not
                # model; a recharging drone within DOCK_ACCEPT_M may take it. (Goals are reached
                # to within STOP_M, so a 0.8 m test left drones hovering at 0.85 m until empty.)
                if tk and tk['kind'] == 'recharge' and b.try_dock(dog, DOCK_ACCEPT_M if self.sched == 'hybrid'
                                                                  else DOCK_RADIUS):
                    self.counts['recharges'] += 1
                    self.log('docked', robot=n, battery_s=round(b.battery, 1))
                    self.tasks[n] = None
                    dtk = self.tasks.get(DOG)
                    if dtk and dtk['kind'] == 'meet' and not any(
                            self.tasks.get(d) and self.tasks[d]['kind'] == 'recharge' and self.tasks[d].get('meet')
                            for d in DRONES):
                        self.set_task(DOG, None, why='rendezvous done')
            touched = self.world.sense(self.t)
            needy = any(self.tasks[n] is None and self.spin_left[n] <= 0 and self.active(n) for n in self.names)
            if step % FIELD_EVERY == 0 or (needy and step - self._last_refresh >= 2):
                self.refresh()
                self._last_refresh = step
            if step % 50 == 0:
                self.explored.append((round(self.t, 1), round(self.world.map.known_area_m2(), 1),
                                      round(self.world.map.searched_area_m2(), 1)))

            for _rt, _s, kind, pl in self.due():
                if kind == 'brain_reply':
                    self.apply_plan(pl['reply'])
                elif kind == 'jev_order':
                    self.rank_inflight = False
                    if self.pt:
                        self.fp.update(pl['p'])
                    else:
                        self.order = [f for f in pl['order'] if f in self.registry.active]
                        self.order_t = self.t
                    self.reassign_explorers()
                elif kind == 'frontier_choice':
                    n = pl['robot']
                    self.frontier_inflight[n] = False
                    if self.tasks.get(n) is None and pl['fid'] and self.active(n):
                        task = self.frontier_task(n, pl['fid'], f"local_{self.dec[self.role(n)].kind}")
                        if task:
                            self.counts['frontier_fallbacks'] += 1
                            self.set_task(n, task, why='local choice while brain pending')
                elif kind == 'verify_reply':
                    tr = self.world.tracker.tracks[pl['track']]
                    tr.vlm.append(pl['answer'])
                    self.cand[pl['track']]['check_pending'] = False
                    self.log('verify_reply', robot=pl['robot'], track=pl['track'],
                             answer={k: pl['answer'][k] for k in ('category', 'clear', 'description')},
                             track_was_real=self.world.track_is_real(tr))
                    self.decide(pl['robot'], tr, 'second_opinion_returned')
                elif kind == 'candidate_action':
                    n = pl['robot']
                    self.inflight[n] = False
                    if pl.get('p') is not None and pl['track'] in self.cand:
                        self.cand[pl['track']]['p'] = pl['p']    # known to the team once the reply lands
                    self.act(n, pl['track'], pl['action'])
                    # A second opinion that came back while this call was in flight was dropped
                    # without a decision (59 of 469 in one run): decide on it now.
                    if self.deferred.get(n) and not self.result:
                        tid, why = self.deferred.pop(n)
                        if tid in self.cand:
                            self.decide(n, self.world.tracker.tracks[tid], why)
                if self.result:
                    break
            if self.result:
                break

            # Battery safety is code, not policy.
            for n in DRONES:
                b_ = self.body(n)
                late = (self.sched == 'hybrid' and b_.state == 'airborne' and self._home.get(n) is not None and
                        b_.battery <= 1.15 * self._home[n] + ag.RESERVE_S)
                if ag.needs_recharge(self.world, n) or late:
                    tk = self.tasks.get(n)
                    if not (tk and tk['kind'] == 'recharge'):
                        self.counts['safety_recharges'] += 1
                        if tk and tk.get('target') in self.cand and self.cand[tk['target']]['handled_by'] == n:
                            self.cand[tk['target']]['handled_by'] = None
                        self.set_task(n, {'kind': 'recharge', 'target': None, 'goal': (dog.x, dog.z),
                                          'source': 'safety', 'since': self.t}, why='battery reserve')

            self.maintain_tasks()

            for tr in {id(x): x for x in touched}.values():
                if tr.label != self.target:
                    continue
                if tr.track_id not in self.cand:
                    self.cand[tr.track_id] = {'status': 'open', 'handled_by': None, 'check_pending': False,
                                              'dog_requested': False, 'dog_attempts': 0}
                    self.log('candidate_new', track=tr.track_id, summary=tr.summary(self.t),
                             real=self.world.track_is_real(tr))
                    if self.sched == 'hybrid' and self.brain is not None:
                        self.brain_wanted = True
                    if self.allocation == 'geometric':
                        self.geometric_dispatch()
                for n in sorted({v[1] for v in tr.views if v[0] >= self.t - 1e-6}):
                    self.decide(n, tr, 'new_view')

            self._idle = []
            for n in self.names:
                self.assign_if_needed(n)
            if self._idle:
                self.allocate(self._idle)
            self.maybe_brain()

            # The dog stands still while a drone makes its final approach to dock:
            # chasing a walking dog was one way recalls ended in a lost drone.
            dog_waits = any(self.tasks.get(d) and self.tasks[d]['kind'] == 'recharge' and
                            not self.tasks[d].get('meet') and       # at a rendezvous the dog walks in
                            self.body(d).status != 'blocked' and    # a blocked drone must not freeze the dog
                            self.body(d).state == 'airborne' and
                            math.hypot(self.body(d).x - dog.x, self.body(d).z - dog.z) < DOG_WAIT_M
                            for d in DRONES)
            for n in self.names:
                b = self.body(n)
                if not self.active(n):
                    continue
                if n == DOG and dog_waits:
                    continue
                if self.spin_left[n] > 0:
                    b.spin()
                    self.spin_left[n] -= CONTROL_DT
                elif self.tasks[n] and self.tasks[n]['kind'] != 'hold':
                    b.step(self.t)
                elif not self.usable(n):
                    b.spin()
            if self.video is not None and step % 3 == 0:
                self.video.append_data(np.asarray(self.frame()))
            if self._events and step % 10 == 0:
                self.log('poses', poses={n: [round(self.body(n).x, 2), round(self.body(n).z, 2),
                                             round(self.body(n).yaw, 2), self.body(n).state,
                                             round(min(self.body(n).battery, 999), 1)] for n in self.names})

        if not self.result:
            self.result = {'outcome': 'timeout', 't': round(self.t, 2), 'success': False}
        tk = self.tasks.get(DOG)
        if tk and tk['kind'] == 'verify':
            self.close_trip(tk, 'mission ended')
        return self.finish(time.perf_counter() - wall0)

    def maintain_tasks(self):
        dog = self.body(DOG)
        for n in self.names:
            tk, b = self.tasks[n], self.body(n)
            if not tk or not self.active(n):
                continue
            kind = tk['kind']
            if kind == 'explore':
                live = self.registry.active.get(tk['target'])
                if live is not None and n in live['goal']:
                    g = live['goal'][n]
                    if math.hypot(g[0] - tk['goal'][0], g[1] - tk['goal'][1]) > 0.5:
                        tk['goal'] = g
                        b.set_goal(g, self.t)
                if tk['target'] not in self.registry.active:
                    self.set_task(n, None, why=f"{tk['target']} explored")
                elif b.status == 'arrived':
                    m = self.world.map
                    m.exhaust(b.x, b.z)
                    # Also the frontier cells this goal was chosen for, and do not hand the same
                    # frontier straight back: the part beyond the 0.6 m disk kept the same goal and a
                    # drone 'reached' it every tick, 1,306 times in one run (62 of 85 episodes).
                    if live is not None:
                        ii, jj = live['cells']
                        wx, wz = m.world(ii, jj)
                        near = np.hypot(np.asarray(wx) - tk['goal'][0], np.asarray(wz) - tk['goal'][1]) <= 1.0
                        m.dead[ii[near], jj[near]] = True
                        m._cache = None
                    self.blacklist[n][tk['target']] = self.t + BLACKLIST_S
                    self.set_task(n, None, why=f"reached {tk['target']}")
                elif b.status == 'blocked':
                    self.blacklist[n][tk['target']] = self.t + BLACKLIST_S
                    self.set_task(n, None, why=f"blocked on {tk['target']} ({getattr(b, 'hold', None)})")
            elif kind == 'recharge':
                # Aim at the nearest cell the drone can actually fly to within
                # docking range of the dog: the dog's own cell is often closed in
                # the (more inflated) air layer, and every 82sE5b5pLXE battery-out
                # in development was a drone already on its way home that never arrived.
                dock = tk.get('dock')
                stale = dock is None or (not tk.get('meet') and
                                         math.hypot(dock[0] - dog.x, dock[1] - dog.z) > DOCK_RADIUS)
                # Progress watchdog: the drone's own forward-depth safety stop can halt it
                # while the planner still has a path, so "blocked" alone missed stalls.
                gd = math.hypot(b.x - tk['goal'][0], b.z - tk['goal'][1]) if tk.get('goal') else 0.0
                pt, pd = tk.get('prog', (self.t, gd))
                if gd < pd - 0.3:
                    tk['prog'] = (self.t, gd)
                elif 'prog' not in tk:
                    tk['prog'] = (self.t, gd)
                stalled = self.t - pt > 4.0 and gd > DOCK_ACCEPT_M
                stuck = (b.status == 'blocked' or stalled) and self.t - tk.get('dock_t', -math.inf) > 2.0
                if stuck:
                    tk.pop('prog', None)
                if stale or stuck:
                    # Blocked on the way to a dock cell: meet the dog instead.
                    dock = None if (stuck and self.sched == 'hybrid') else self.dock_cell(n, dog)
                    tk['meet'] = False
                    if dock is None and self.sched == 'hybrid':
                        # The dog stands where this drone cannot fly (e.g. a low-clearance
                        # corridor): meet at the point both can reach soonest (UAV-UGV
                        # rendezvous); the dog leaves exploring for it, not a verify trip.
                        dock = self.rendezvous(n)
                        if dock is not None:
                            tk['meet'] = True
                            cur = self.tasks.get(DOG)
                            if not (cur and cur['kind'] == 'verify') and not (
                                    cur and cur['kind'] == 'meet' and math.hypot(cur['goal'][0] - dock[0],
                                                                                 cur['goal'][1] - dock[1]) < 1.0):
                                self.set_task(DOG, {'kind': 'meet', 'target': None, 'goal': dock,
                                                    'source': 'rendezvous', 'since': self.t},
                                              why=f'rendezvous with {n} to recharge')
                    dock = dock or (dog.x, dog.z)
                    tk['dock'], tk['goal'], tk['dock_t'] = dock, dock, self.t
                    b.set_goal(None, self.t)
                    b.set_goal(dock, self.t)
            elif kind == 'meet':
                waiting = any(self.tasks.get(d) and self.tasks[d]['kind'] == 'recharge' and self.tasks[d].get('meet')
                              and self.body(d).state == 'airborne' for d in DRONES)
                if not waiting or self.t - tk['since'] > MEET_TIMEOUT_S:
                    self.set_task(n, None, why='rendezvous over' if waiting else 'no drone waiting')
            elif kind in ('inspect', 'verify'):
                tr = self.world.tracker.tracks[tk['target']]
                st = self.cand[tk['target']]
                if kind == 'inspect':
                    limit = INSPECT_LIMIT_S
                else:   # the dog walks at 1 m/s; allow the straight-line trip twice over, plus 25 s
                    o = tk.get('trip_origin', (b.x, b.z))
                    limit = 25.0 + 2.0 * math.hypot(tr.position[0] - o[0], tr.position[2] - o[1])
                if st['status'] != 'open':
                    self.set_task(n, None, why='candidate closed')
                elif kind == 'verify' and n == DOG and tk.get('scout') and \
                        (b.status in ('arrived', 'blocked') or self.t - tk['scout_t'] > 2.0):
                    # Walking toward a candidate with no standoff: once the floor round it is mapped, stand there.
                    tk['scout_t'] = self.t
                    goal = self.standoff(DOG, tr)
                    if goal:
                        tk.pop('scout'); tk['goal'] = goal; st['last_goal'] = goal
                        b.set_goal(goal, self.t)
                        self.log('scout_standoff', robot=n, track=tr.track_id, goal=[round(v, 2) for v in goal])
                    elif b.status in ('arrived', 'blocked'):
                        nxt = self.scout_goal(tr)
                        if nxt:
                            tk['goal'] = nxt
                            b.set_goal(None, self.t); b.set_goal(nxt, self.t)
                        else:
                            self.set_task(n, None, why='scout: no way closer')   # the trip counted as an attempt
                elif kind == 'verify' and n == DOG and st.get('final_approach'):
                    if b.status in ('arrived', 'blocked') or self.t - st['final_approach']['t0'] > st['final_approach']['limit_s']:
                        self.declare(n, tr.track_id, tr)
                elif self.t - tk['since'] > limit:
                    st['handled_by'] = None if st['handled_by'] == n else st['handled_by']
                    if kind == 'inspect':
                        st.setdefault('inspected_by', {})[n] = True
                    self.set_task(n, None, why=f'{kind} timed out')
                elif self.sched == 'hybrid' and kind == 'verify' and n == DOG and b.status not in ('arrived', 'blocked') \
                        and self._should_look(n, tr, st, en_route=True):
                    # On the way in: a clear view from a couple of metres frames the whole
                    # object; at the 0.5 m standoff a 0.5 m-high camera often cannot.
                    st['look_t'] = self.t
                    self.act(n, tr.track_id, 'second_opinion')
                elif b.status in ('arrived', 'blocked') and kind == 'inspect' and \
                        (n, tr.track_id) in self.arrived_at and \
                        self.t - self.arrived_at[(n, tr.track_id)] > NO_VIEW_GIVE_UP_S and \
                        tr.last_crop.get(n, (-math.inf,))[0] < self.arrived_at[(n, tr.track_id)]:
                    # At the standoff but the detector never fired on it from here.
                    st['handled_by'] = None
                    st.setdefault('inspected_by', {})[n] = True
                    self.set_task(n, None, why='inspect: no view from standoff')
                elif kind == 'verify' and n == DOG and self.spin_left[n] > 0:
                    pass            # looking around: no reposition or give-up mid-turn
                elif b.status in ('arrived', 'blocked') and kind == 'verify' and n == DOG and \
                        self.sched == 'hybrid' and self._should_look_around(tr, st):
                    # Its last two looks at the estimate saw no target and its own detector has
                    # nothing fresh: the estimate is off (CrMo8WxCyVb:12, a vase placed on the wall
                    # behind it). Turn once in place with a dense detector pass; the looks restart.
                    st['scan_n'] = st.get('scan_n', 0) + 1
                    spin = 2 * math.pi / 2.0                    # dog Body.spin is 2 rad/s
                    self.spin_left[n] = spin
                    self.world.dense_until = self.t + spin + 0.2
                    self.arrived_at[(n, tr.track_id)] = self.t + spin     # reposition timer after the turn
                    st['own_reset'] = sum(1 for c in tr.vlm if c.get('by') == DOG)
                    self.log('look_around', robot=n, track=tr.track_id, n=st['scan_n'])
                elif b.status in ('arrived', 'blocked') and kind == 'verify' and n == DOG and \
                        not st['check_pending'] and st.get('p', 1.0) < ag.DOG_REQUEST_GATE and \
                        sum(1 for c in tr.vlm if c.get('by') == DOG) - st.get('own_reset', 0) >= 3:
                    # Its own looks are used up and the evidence says no: release the dog for good
                    # (the only exit used to be the trip timeout, twice: 90 s on one false candidate).
                    st['dog_attempts'] = MAX_DOG_ATTEMPTS
                    self.set_task(n, None, why=f"{tr.track_id} unsettled (P {st.get('p', 0):.2f})")
                elif b.status in ('arrived', 'blocked') and kind == 'verify' and n == DOG and \
                        self.t - self.arrived_at.get((n, tr.track_id), math.inf) > REPOSITION_S and \
                        not st['check_pending'] and self.reposition(tk, tr, st, b):
                    pass
                elif b.status in ('arrived', 'blocked'):
                    loc = self.world.aim_point(n, tr)
                    b.face_target = (float(loc[0]), float(loc[2]))
                    if (n, tr.track_id) not in self.arrived_at:
                        self.arrived_at[(n, tr.track_id)] = self.t
                        self.decide(n, tr, 'arrived')
                    else:
                        self.decide(n, tr, 'new_view')
                    # At the candidate the dog always takes its own look until it has one
                    # clear answer (or three tries): declaring requires it.
                    if self.sched == 'hybrid' and kind == 'verify' and n == DOG and self._should_look(n, tr, st):
                        st['look_t'] = self.t
                        self.act(n, tr.track_id, 'second_opinion')
                    if kind == 'inspect' and tr.vlm and not st['check_pending'] and \
                            self.t - self.arrived_at[(n, tr.track_id)] > 4.0:
                        st['handled_by'] = None
                        st.setdefault('inspected_by', {})[n] = True
                        self.set_task(n, None, why='inspected')

    # ------------------------------------------------------------ output
    def finish(self, wall_s):
        w = self.world
        success = self.result.get('outcome') == 'success'
        dog = self.body(DOG)
        real = [tid for tid in self.cand if w.track_is_real(w.tracker.tracks[tid])]
        first_real = min((w.tracker.tracks[t].first_t for t in real), default=None)
        empty = [tp for tp in self.trips if tp['empty_run']]

        def pct(xs, q):
            return round(float(np.percentile(xs, q)), 3) if xs else None
        summary = {
            'episode_id': self.ep['episode_id'], 'target': self.target, 'allocation': self.allocation,
            'frontier_cost': 'pt' if self.pt else 'rank', 'dog_explores': DOG_EXPLORES,
            'drone_local': self.dec['drone'].kind, 'dog_local': self.dec['dog'].kind, 'clock': self.clock,
            **self.result, 'success': success, 'time_s': self.result['t'],
            'spl': round(self.ep['shortest_geodesic_m'] / max(dog.travelled, self.ep['shortest_geodesic_m']), 4)
            if success else 0.0,
            'dog_path_m': round(dog.travelled, 2), 'dog_collisions': self.world.collision_log[:50],
            **({'objectnav_success': bool(self.result.get('objectnav_success')),
                'objectnav_success_012': bool(self.result.get('objectnav_success_012')),
                'objectnav_dtg_m': self.world.viewpoint_distance(DOG),
                'objectnav_dtg_start_navigable': getattr(self.world, 'dtg_start_navigable', None),
                'objectnav_spl': round(self.ep['shortest_geodesic_m'] /
                                       max(dog.travelled, self.ep['shortest_geodesic_m']), 4)
                if self.result.get('objectnav_success') else 0.0,
                'objectnav_spl_012': round(self.ep['shortest_geodesic_m'] /
                                           max(dog.travelled, self.ep['shortest_geodesic_m']), 4)
                if self.result.get('objectnav_success_012') else 0.0}
               if self.world._viewpoints is not None else {}),
            'drone_path_m': {n: round(self.body(n).travelled, 2) for n in DRONES},
            'drone_flight_s': {n: round(self.body(n).flight_s, 1) for n in DRONES},
            'dog_trips': len(self.trips), 'dog_empty_runs': len(empty),
            'dog_empty_run_m': round(sum(tp['walked_m'] for tp in empty), 2),
            'dog_trips_to_false': sum(tp['wasted_on_false'] for tp in self.trips),
            'dog_m_to_false': round(sum(tp['walked_m'] for tp in self.trips if tp['wasted_on_false']), 2),
            'trips': self.trips,
            'explored_m2_final': round(w.map.known_area_m2(), 1),
            'searched_m2_final': round(w.map.searched_area_m2(), 1), 'explored_curve': self.explored,
            'candidates': len(self.cand), 'real_candidates': len(real), 'first_real_candidate_seen_s': first_real,
            'counts': self.counts,
            'latency_s': {k: {'n': len(v), 'p50': pct(v, 50), 'p95': pct(v, 95), 'total': round(float(sum(v)), 2)}
                          for k, v in self.latency.items()},
            'detector_calls': w.detector_calls, 'wall_s': round(wall_s, 1)}
        if self.out:
            (self.out / 'summary.json').write_text(json.dumps(summary, indent=2, default=_jsonable))
            if self.world.loc_eval:
                (self.out / 'localization_eval.json').write_text(json.dumps(self.world.loc_eval, default=_jsonable))
        if self._events:
            self._events.close()
        if self.video is not None:
            self.video.close()
        w.close()
        return summary

    def frame(self):
        from PIL import Image, ImageDraw
        W, H = 1280, 736
        canvas = Image.new('RGB', (W, H), (18, 18, 22))
        dr = ImageDraw.Draw(canvas)
        f16, f13, f12 = tb.font(16), tb.font(13), tb.font(12)
        for k, n in enumerate(self.names):
            b = self.body(n)
            rgb = self.world.frames[k][f'{n}_rgb'][..., :3]
            im = Image.fromarray(np.asarray(rgb, np.uint8)).resize((232, 232))
            x0 = 10 + k * 240
            canvas.paste(im, (x0, 10))
            col = board.COLOUR.get(n, (200, 200, 200))
            dr.rectangle([x0, 10, x0 + 231, 241], outline=col, width=3)
            dr.rectangle([x0, 10, x0 + 231, 30], fill=(0, 0, 0))
            label = f"{n} {board.task_text(self.tasks.get(n))}"
            if b.aerial:
                label += f"  {max(b.battery, 0):.0f}s" + ('' if b.state == 'airborne' else f' {b.state}')
            dr.text((x0 + 5, 12), label[:34], fill=col, font=f12)
        cands = self.candidates_view()
        canvas.paste(board.render(self.world, self.registry, self.tasks, cands, size=470, legend=False), (10, 256))
        x0, y = 500, 256
        mode = (f"alloc={self.allocation} drone={self.dec['drone'].kind} dog={self.dec['dog'].kind} "
                f"clock={self.clock}")
        lines = [(f"Find a {phrase(self.target)}   t = {self.t:5.1f} / {self.mission_s:.0f} s", f16, (255, 255, 255)),
                 (f"{self.ep['episode_id']}   {mode}", f12, (170, 170, 170)),
                 (f"explored {self.world.map.known_area_m2():.0f} m2  dog trips {len(self.trips)}  "
                  f"empty {sum(tp['empty_run'] for tp in self.trips)}", f12, (170, 170, 170)),
                 ('', f12, (0, 0, 0)), ('SCHEDULER', f13, (255, 200, 80)), (self.hud['brain'][:90], f12, (230, 230, 230))]
        for n in self.names:
            if n in self.hud['why']:
                lines.append((f"  {n} {self.hud['why'][n]}"[:100], f12, (200, 200, 200)))
        lines += [('', f12, (0, 0, 0)), ('LOCAL', f13, (120, 220, 255)), (self.hud['local'][:95], f12, (230, 230, 230)),
                  ('', f12, (0, 0, 0)), (f'CANDIDATES ({phrase(self.target)})', f13, (255, 120, 120))]
        for c in cands[-6:]:
            ops = ', '.join(o['category'] + ('' if o['clear'] else '?') for o in c['second_opinions']) or '-'
            lines.append((f"  {c['id']} v={c['views']} s={c['max_score']:.2f} {c['status']}"
                          f"{' dog-req' if c['dog_requested'] else ''}  2nd: {ops}"[:100], f12, (220, 220, 220)))
        if self.result:
            lines += [('', f12, (0, 0, 0)), (f"RESULT: {self.result['outcome']}", f16,
                                             (120, 255, 120) if self.result.get('success') else (255, 90, 90))]
        for text, fnt, col in lines:
            dr.text((x0, y), text, fill=col, font=fnt)
            y += 18
        return canvas


def _view(task):
    if not task:
        return None
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in task.items() if k != 'trip'}


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, bytes):
        return f'<{len(o)} bytes>'
    return str(o)


SOURCES = ('team_run.py', 'team_world.py', 'team_agents.py', 'team_board.py', 'aerial_map.py',
           'observed_map.py', 'tactical_board.py', 'search_agents.py', 'search_world.py',
           'verifier_prompt.py', 'jev.py', 'mask_localize.py', 'perception.py', 'jev_rank.py', 'vocabulary.py',
           'team_experience_v3.py')


def load_detector():
    from perception import OpenVocabDetector
    from vocabulary import VOCABULARY
    return OpenVocabDetector(VOCABULARY, max_detections=12, backend='gdino', threshold=0.25)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--episodes', required=True)
    ap.add_argument('--episode-id', required=True)
    ap.add_argument('--allocation', choices=['geometric', 'brain', 'direct', 'jev'], required=True)
    ap.add_argument('--dog-explores', choices=['yes', 'no'], default='yes',
                    help='no: the legged robot takes no frontiers and moves only for a gated candidate')
    ap.add_argument('--frontier-cost', choices=['rank', 'pt'], default='rank',
                    help='brain allocation: travel + RANK_S * rank (rank), or travel / judge P(target beyond) (pt)')
    ap.add_argument('--drone-local', choices=['rule', 'jev', 'vlm'], required=True)
    ap.add_argument('--dog-local', choices=['rule', 'jev', 'vlm'], required=True)
    ap.add_argument('--clock', choices=['charged', 'free'], default='charged')
    ap.add_argument('--out', required=True)
    ap.add_argument('--video', action='store_true')
    ap.add_argument('--mission-s', type=float, default=MISSION_S)
    ap.add_argument('--drones-may-declare', action='store_true')
    ap.add_argument('--allow-paid-inference', action='store_true')
    ap.add_argument('--experience', help='case memory (team_experience_v3.py build/freeze); omit for none')
    ap.add_argument('--experience-learn', action='store_true',
                    help='v3 store, practice episodes only: add this run\'s declaration to the store')
    ap.add_argument('--scheduler', choices=['full', 'hybrid'], default='hybrid',
                    help='full: brain assigns every task incl. frontiers; hybrid: brain ranks frontiers and assigns '
                         'semantic tasks, a Hungarian assignment does the geometry (also for geometric allocation)')
    # mask by default: the box's depth mixes the object with what stands in front of it (a table before a sofa)
    # or behind it; over 708 views of real targets (HM3D val, 8 episodes, --localize eval) the box put 84% (drones)
    # / 93% (dog) within 0.3 m of the object, the SAM mask 98% / 100%, worse by > 0.1 m in 2 views.
    ap.add_argument('--localize', choices=['box', 'eval', 'mask'], default='mask',
                    help='mask: place target detections by SAM mask (default); box: box foreground depth; '
                         'eval: box placement but log SAM mask estimates too')
    a = ap.parse_args()
    doc = json.loads(Path(a.episodes).read_text())
    experience = json.loads(Path(a.experience).read_text()) if a.experience else None
    ep = next(e for e in doc['episodes'] if e['episode_id'] == a.episode_id)
    if experience:
        import team_experience_v3
        team_experience_v3.install(a.experience, ep, learn=a.experience_learn)
    elif a.experience_learn:
        raise SystemExit('--experience-learn needs a case-memory store (team_experience_v3.py build)')
    global DOG_EXPLORES
    DOG_EXPLORES = a.dog_explores == 'yes'
    needs_jev = 'jev' in (a.drone_local, a.dog_local, a.allocation) or a.frontier_cost == 'pt'   # pt: the judge scores frontiers
    if needs_jev and not a.allow_paid_inference:
        raise SystemExit('a jev decider needs --allow-paid-inference')
    if a.allocation in ('direct', 'jev') and a.scheduler != 'hybrid':
        raise SystemExit('--allocation direct/jev needs --scheduler hybrid')
    from jev import JevClient
    jev = JevClient() if needs_jev else None
    server = VLMServer()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).parent
    roles = ('dog', 'drone') if a.drones_may_declare else ('dog',)
    (out / 'manifest.json').write_text(json.dumps({
        'episode': ep, 'allocation': a.allocation, 'drone_local': a.drone_local, 'dog_local': a.dog_local,
        'clock': a.clock, 'mission_s': a.mission_s, 'declare_roles': roles, 'localize': a.localize, 'scheduler': a.scheduler,
        'experience': a.experience, 'experience_learn': a.experience_learn, 'experience_sha256': (hashlib.sha256(Path(a.experience).read_bytes()).hexdigest()
                                                          if a.experience else None),
        'thresholds': {k: getattr(ag, k) for k in ('DECLARE_RANGE', 'DECLARE_GATE', 'REJECT_GATE',
                                                    'DOG_REQUEST_GATE', 'RESERVE_S')},
        'source_sha256': {s: hashlib.sha256((here / s).read_bytes()).hexdigest() for s in SOURCES}}, indent=2))
    localizer = None
    if a.localize != 'box':
        from mask_localize import MaskLocalizer
        localizer = MaskLocalizer()
    r = Runner(ep, a.allocation, a.drone_local, a.dog_local, load_detector(), server, jev, out_dir=out,
               video=a.video, clock=a.clock, mission_s=a.mission_s, declare_roles=roles, experience=experience,
               localizer=localizer, localize_mode='mask' if a.localize == 'mask' else 'box',
               scheduler=a.scheduler, frontier_cost=a.frontier_cost)
    s = r.run()
    print(json.dumps({k: s[k] for k in ('episode_id', 'allocation', 'drone_local', 'dog_local', 'clock',
                                        'outcome', 'time_s', 'dog_trips', 'dog_empty_runs', 'dog_path_m',
                                        'counts', 'latency_s', 'wall_s')}, indent=1))


if __name__ == '__main__':
    main()

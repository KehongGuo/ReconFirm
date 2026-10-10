"""Multi-robot object search in an unknown building: the world side.

Everything a policy may touch lives on `SearchWorld.observable()`: the map the
team has built from its own depth frames, the landmark tracks its detector has
produced, and each robot's own pose. Everything else here -- the navmesh, the
semantic annotations, geodesic distances -- is either physics (collision) or
evaluation (scoring, episode generation) and is never handed to a policy.

TASK (declared before any method was run)
  Three ground robots start together at one point of a development building and
  search for one object category. Any robot may declare "found"; the episode then
  ends. It is a success iff the declaring robot stands within SUCCESS_DISTANCE of
  a real instance of the category, measured horizontally to the object's oriented
  bounding box, on the starting floor. A wrong declaration is a failure, as is
  running out of MISSION_SECONDS. Categories: the five HM3D-ObjectNav classes the
  wide detector vocabulary contains, minus chair (present in every building, so
  finding one says little).

KNOWN LIMITS, stated rather than hidden: poses are known (no SLAM), the team
shares a frame, search is single-floor, detector compute time is not charged
against the mission clock (model and brain latency are).
"""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TARGETS = ('bed', 'toilet', 'sofa', 'tv_monitor', 'plant')
SUCCESS_DISTANCE = 1.0
MIN_START_GEODESIC = 3.0
MISSION_SECONDS = 180.0
CONTROL_DT = 0.1
N_ROBOTS = 3
RESOLUTION = 384
HFOV = 79.0
CAMERA_HEIGHT = 0.88
DETECT_EVERY = 5          # ticks between detector passes, per robot
MAP_EVERY = 2             # ticks between depth integrations, per robot
DETECT_RANGE = 6.0        # metres; beyond this a box's position is not trusted
TRACK_RADIUS = 1.0        # same-label detections this close join one track


def searcher_team(n=N_ROBOTS, resolution=RESOLUTION):
    from team_sim import RobotSpec
    return [RobotSpec(name=f'r{i}', kind='ground', sensor_height=CAMERA_HEIGHT, hfov=HFOV,
                      resolution=(resolution, resolution), max_speed=1.0)
            for i in range(n)]


def scene_ref(scene_id):
    from scenes import mp3d_scenes, hm3d_scene
    ref = next((r for r in mp3d_scenes() if Path(r.scene_id).stem == scene_id), None) or hm3d_scene(scene_id)
    if ref is None:
        raise SystemExit(f'Scene not found in the local MP3D or HM3D install: {scene_id}')
    return ref


# ---------------------------------------------------------------- truth (evaluation only)
def target_instances(sim, category, floor_y):
    out = []
    for o in sim.sim.semantic_scene.objects:
        if o is None or o.category is None or o.category.name() != category:
            continue
        c = np.asarray(o.obb.center, dtype=np.float64)
        if not (floor_y - 0.3 <= c[1] <= floor_y + 2.2):
            continue
        out.append({'instance_id': int(o.semantic_id), 'obb': o.obb, 'center': c.tolist()})
    return out


def horizontal_distance(obb, x, z):
    import magnum as mn
    c = obb.center
    return float(obb.distance(mn.Vector3(float(x), float(c[1]), float(z))))


def success_viewpoints(sim, obb, floor_y, island):
    """Navigable points on the start floor within SUCCESS_DISTANCE of the object."""
    pf = sim.pathfinder
    c = np.asarray(obb.center)
    r = float(np.max(np.asarray(obb.half_extents))) + SUCCESS_DISTANCE
    pts = []
    for dx in np.arange(-r, r + 1e-6, 0.25):
        for dz in np.arange(-r, r + 1e-6, 0.25):
            p = np.array([c[0] + dx, floor_y, c[2] + dz], dtype=np.float32)
            s = np.asarray(pf.snap_point(p))
            if np.any(np.isnan(s)) or abs(s[1] - floor_y) > 0.3 or int(pf.get_island(s)) != island:
                continue
            if horizontal_distance(obb, s[0], s[2]) <= SUCCESS_DISTANCE:
                pts.append(s)
    return pts


def geodesic(sim, a, b):
    import habitat_sim
    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(a, dtype=np.float32)
    path.requested_end = np.asarray(b, dtype=np.float32)
    if not sim.pathfinder.find_path(path) or not np.isfinite(path.geodesic_distance):
        return float('inf')
    return float(path.geodesic_distance)


# ---------------------------------------------------------------- episodes
def make_episode(scene_id, seed):
    """One frozen search episode for a building. Uses the navmesh -- task setup,
    like any benchmark's episode generator -- and nothing it computes reaches a
    policy except the start poses and the target category."""
    from team_sim import TeamSim
    ref = scene_ref(scene_id)
    sim = TeamSim(ref.scene_id, ref.dataset_config, team=searcher_team(), control_dt=CONTROL_DT)
    try:
        sim.build_navmeshes()
        pf = sim.pathfinder
        island = sim.main_island('ground')
        rng = random.Random(f'{scene_id}:{seed}')
        pf.seed(rng.randrange(1 << 30))
        for _attempt in range(60):
            start = np.asarray(pf.get_random_navigable_point(), dtype=np.float32)
            if np.any(np.isnan(start)) or int(pf.get_island(start)) != island:
                continue
            floor_y = float(start[1])
            starts = []
            for k in range(N_ROBOTS):
                ang = 2 * math.pi * k / N_ROBOTS
                p = np.asarray(pf.snap_point(start + np.array(
                    [0.45 * math.cos(ang), 0, 0.45 * math.sin(ang)], np.float32)))
                if np.any(np.isnan(p)) or abs(p[1] - floor_y) > 0.2 or int(pf.get_island(p)) != island:
                    break
                starts.append([float(v) for v in p])
            if len(starts) < N_ROBOTS:
                continue
            eligible = {}
            for cat in TARGETS:
                best = float('inf')
                for inst in target_instances(sim, cat, floor_y):
                    for vp in success_viewpoints(sim, inst['obb'], floor_y, island):
                        best = min(best, geodesic(sim, start, vp))
                if np.isfinite(best) and best >= MIN_START_GEODESIC:
                    eligible[cat] = best
            if not eligible:
                continue
            target = rng.choice(sorted(eligible))
            return {'scene': scene_id, 'seed': seed, 'target': target,
                    'floor_y': floor_y, 'start': [float(v) for v in start],
                    'robot_starts': starts,
                    'robot_yaws': [2 * math.pi * k / N_ROBOTS for k in range(N_ROBOTS)],
                    'shortest_geodesic_m': round(eligible[target], 3),
                    'eligible_targets': {k: round(v, 3) for k, v in eligible.items()}}
        return None
    finally:
        sim.close()


# ---------------------------------------------------------------- perception tracks
@dataclass
class Track:
    track_id: str
    label: str
    positions: list = field(default_factory=list)
    scores: list = field(default_factory=list)
    robots: set = field(default_factory=set)
    views: list = field(default_factory=list)       # (t, robot, distance, bearing)
    first_t: float = 0.0
    last_t: float = 0.0
    best_crop: bytes | None = None
    best_crop_score: float = -1.0
    last_crop: dict = field(default_factory=dict)   # robot -> (t, bytes)
    vlm: list = field(default_factory=list)         # verification answers
    truth: Counter = field(default_factory=Counter)  # evaluation only

    @property
    def position(self):
        return np.median(np.asarray(self.positions), axis=0)

    def summary(self, now, from_xz=None):
        p = self.position
        bearings = [v[3] for v in self.views]
        span = 0.0
        if len(bearings) > 1:
            span = max(abs((a - b + math.pi) % (2 * math.pi) - math.pi)
                       for i, a in enumerate(bearings) for b in bearings[i + 1:])
        out = {'id': self.track_id, 'label': self.label,
               'views': len(self.views), 'robots': len(self.robots),
               'max_score': round(max(self.scores), 3),
               'mean_score': round(float(np.mean(self.scores)), 3),
               'closest_view_m': round(min(v[2] for v in self.views), 2),
               'bearing_span_rad': round(span, 2),
               'first_seen_s': round(self.first_t, 1), 'last_seen_s': round(self.last_t, 1),
               'xz': [round(float(p[0]), 2), round(float(p[2]), 2)]}
        if self.vlm:
            out['vlm_checks'] = [{k: v[k] for k in ('category', 'clear', 'description', 't')}
                                 for v in self.vlm]
        return out


class Tracker:
    """Same-label detections within TRACK_RADIUS join one track. Deliberately
    simple and declared up front; association is not what this study is about."""

    def __init__(self):
        self.tracks: dict[str, Track] = {}
        self._n = 0

    def add(self, label, position, score, robot, t, distance, bearing, crop, truth_ids):
        best, best_d = None, TRACK_RADIUS
        for tr in self.tracks.values():
            if tr.label != label:
                continue
            d = float(np.hypot(*(tr.position - position)[[0, 2]]))
            if d <= best_d:
                best, best_d = tr, d
        if best is None:
            self._n += 1
            best = Track(track_id=f'T{self._n}', label=label, first_t=t)
            self.tracks[best.track_id] = best
        best.positions.append(np.asarray(position, dtype=np.float64))
        best.scores.append(float(score))
        best.robots.add(robot)
        best.views.append((t, robot, float(distance), float(bearing)))
        best.last_t = t
        if crop is not None:
            best.last_crop[robot] = (t, crop)
            if score > best.best_crop_score:
                best.best_crop, best.best_crop_score = crop, float(score)
        for tid in truth_ids:
            best.truth[tid] += 1
        return best

    def near(self, x, z, radius, exclude=None):
        out = []
        for tr in self.tracks.values():
            if tr is exclude:
                continue
            p = tr.position
            if math.hypot(p[0] - x, p[2] - z) <= radius:
                out.append(tr)
        return out


# ---------------------------------------------------------------- motion on the observed map
class Mover:
    """Drive one robot along routes planned on the observed map only.

    `try_step` against the navmesh is collision -- it stops a robot walking
    through a wall -- and is the only thing the navmesh does here.
    """
    TURN_RATE = 2.0
    TURN_THRESH = 0.35
    LOOKAHEAD = 0.6
    REPLAN_S = 1.0

    def __init__(self, world, name):
        self.world, self.name = world, name
        self.goal = None          # (x, z)
        self.path = []
        self.replan_at = -1.0
        self.status = 'idle'      # idle | moving | arrived | blocked
        self.progress_ref = None  # (t, x, z) for stall detection
        self.travelled = 0.0
        self.face_target = None

    def set_goal(self, xz, t):
        if xz is None:
            self.goal, self.path, self.status = None, [], 'idle'
            return
        if self.goal is None or math.hypot(xz[0] - self.goal[0], xz[1] - self.goal[1]) > 0.3:
            self.goal = (float(xz[0]), float(xz[1]))
            self.path, self.replan_at, self.status = [], -1.0, 'moving'
            pos, _ = self.world.sim.get_pose(self.name)
            self.progress_ref = (t, float(pos[0]), float(pos[2]))

    def _plan(self, t):
        pos, _ = self.world.sim.get_pose(self.name)
        field_ = self.world.map.distance_field(pos[0], pos[2])
        gi, gj = self.world.map.cell(self.goal[0], self.goal[1])
        self.path = self.world.map.path_to(field_, gi, gj)
        self.replan_at = t + self.REPLAN_S
        if not self.path:
            self.status = 'blocked'

    def turn_to(self, yaw_goal):
        sim = self.world.sim
        pos, yaw = sim.get_pose(self.name)
        err = (yaw_goal - yaw + math.pi) % (2 * math.pi) - math.pi
        step = max(-self.TURN_RATE * CONTROL_DT, min(self.TURN_RATE * CONTROL_DT, err))
        sim.set_pose(self.name, pos, yaw=yaw + step, lifted=True)
        return abs(err) - abs(step)

    def spin(self):
        sim = self.world.sim
        pos, yaw = sim.get_pose(self.name)
        sim.set_pose(self.name, pos, yaw=yaw + self.TURN_RATE * CONTROL_DT, lifted=True)

    def step(self, t):
        if self.goal is None or self.status in ('idle', 'arrived', 'blocked'):
            if self.face_target is not None:
                pos, _ = self.world.sim.get_pose(self.name)
                d = np.array([self.face_target[0] - pos[0], 0, self.face_target[1] - pos[2]])
                self.turn_to(math.atan2(-d[0], -d[2]))
            return self.status
        sim = self.world.sim
        pos, yaw = sim.get_pose(self.name)
        if math.hypot(self.goal[0] - pos[0], self.goal[1] - pos[2]) < 0.25:
            self.status = 'arrived'
            return self.status
        if t >= self.replan_at or not self.path:
            self._plan(t)
            if self.status == 'blocked':
                return self.status
        while len(self.path) > 1 and math.hypot(self.path[0][0] - pos[0], self.path[0][1] - pos[2]) < self.LOOKAHEAD:
            self.path.pop(0)
        wx, wz = self.path[0]
        d = np.array([wx - pos[0], 0.0, wz - pos[2]])
        want = math.atan2(-d[0], -d[2])
        err = (want - yaw + math.pi) % (2 * math.pi) - math.pi
        if abs(err) > self.TURN_THRESH:
            self.turn_to(want)
        else:
            stride = min(1.0 * CONTROL_DT, float(np.hypot(d[0], d[2])))
            fwd = np.array([-math.sin(yaw), 0.0, -math.cos(yaw)], dtype=np.float32)
            goal = np.asarray(pos, dtype=np.float32) + fwd * stride
            stepped = np.asarray(sim.pathfinder.try_step(np.asarray(pos, np.float32), goal))
            # A NaN step means collision could not be resolved. Standing still is
            # the only safe answer: snapping the raw goal can hop a thin wall.
            goal = np.asarray(pos, np.float32) if np.any(np.isnan(stepped)) else stepped
            new = sim.set_pose(self.name, goal, yaw=want, lifted=True)
            self.travelled += float(np.hypot(new[0] - pos[0], new[2] - pos[2]))
        # Stall: under 0.3 m of progress in 4 s of trying.
        ref = self.progress_ref
        pos, _ = sim.get_pose(self.name)
        if ref is None or math.hypot(pos[0] - ref[1], pos[2] - ref[2]) > 0.3:
            self.progress_ref = (t, float(pos[0]), float(pos[2]))
        elif t - ref[0] > 4.0:
            self.status = 'blocked'
        return self.status


# ---------------------------------------------------------------- the world
class SearchWorld:
    def __init__(self, episode, detector=None, resolution=RESOLUTION):
        from team_sim import TeamSim
        from observed_map import ObservedMap
        self.episode = episode
        ref = scene_ref(episode['scene'])
        self.team = searcher_team(resolution=resolution)
        self.sim = TeamSim(ref.scene_id, ref.dataset_config, team=self.team, control_dt=CONTROL_DT)
        self.sim.build_navmeshes()
        for r, p, yaw in zip(self.team, episode['robot_starts'], episode['robot_yaws']):
            self.sim.set_pose(r.name, np.asarray(p, np.float32), yaw=yaw)
        s = episode['start']
        self.map = ObservedMap((s[0], s[2]), floor_y=episode['floor_y'])
        self.tracker = Tracker()
        self.movers = {r.name: Mover(self, r.name) for r in self.team}
        self.detector = detector
        self.frames = None
        self.tick = 0
        self.trails = {r.name: [] for r in self.team}
        # Evaluation-only truth. Never passed to a policy.
        self._instances = target_instances(self.sim, episode['target'], episode['floor_y'])
        self._categories = {int(o.semantic_id): o.category.name()
                            for o in self.sim.sim.semantic_scene.objects
                            if o is not None and o.category is not None}
        self.detector_calls = 0

    @property
    def names(self):
        return [r.name for r in self.team]

    def pose(self, name):
        pos, yaw = self.sim.get_pose(name)
        return float(pos[0]), float(pos[2]), float(yaw)

    def sense(self, t):
        """Render, fold depth into the map, and run the detector on its schedule.
        Returns the tracks that received a new view this tick."""
        from perception import ground_detections, crop_box
        self.frames = self.sim.observe()
        touched = []
        for k, r in enumerate(self.team):
            pos, yaw = self.sim.get_pose(r.name)
            self.map.stand(pos[0], pos[2], t)
            if self.tick % 5 == 0:
                self.trails[r.name].append((round(float(pos[0]), 2), round(float(pos[2]), 2)))
            if (self.tick + k) % MAP_EVERY == 0:
                eye, R = self.sim.camera_transform(r.name)
                self.map.integrate(self.frames[k][f'{r.name}_depth'], eye, R, r.hfov, t)
            if self.detector is None or (self.tick + k) % DETECT_EVERY != 0:
                continue
            rgb = self.frames[k][f'{r.name}_rgb']
            depth = self.frames[k][f'{r.name}_depth']
            sem = self.frames[k].get(f'{r.name}_semantic')
            dets = ground_detections(self.detector.detect(rgb), depth, self.sim, r.name)
            self.detector_calls += 1
            for det in dets:
                if det.position is None or det.distance > DETECT_RANGE or det.label == '?':
                    continue
                h = float(det.position[1]) - self.episode['floor_y']
                if not (-0.3 <= h <= 2.4):
                    continue
                dx, dz = det.position[0] - pos[0], det.position[2] - pos[2]
                truth = []
                if sem is not None:
                    x0, y0, x1, y1 = det.box
                    ids, counts = np.unique(np.asarray(sem)[y0:y1, x0:x1], return_counts=True)
                    for i_, c_ in zip(ids, counts):
                        if int(i_) in self._categories and c_ >= 64 and \
                                self._categories[int(i_)] == det.label:
                            truth.append(int(i_))
                tr = self.tracker.add(det.label, np.asarray(det.position), det.score, r.name, t,
                                      det.distance, math.atan2(dx, dz),
                                      crop_box(rgb, det.box), truth)
                touched.append(tr)
        self.tick += 1
        return touched

    # ------------------------------------------------------------ evaluation only
    def score_declaration(self, name):
        x, z, _ = self.pose(name)
        best = min((horizontal_distance(i['obb'], x, z) for i in self._instances), default=float('inf'))
        return {'success': best <= SUCCESS_DISTANCE, 'distance_to_nearest_target_m': round(best, 3)}

    def track_is_real(self, track):
        """Whether any of a track's boxes covered a real instance of its label."""
        return bool(track.truth)

    def close(self):
        self.sim.close()

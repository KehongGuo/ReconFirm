"""The team's world: three drones and one legged robot searching an unknown building.

What differs from the homogeneous ground-robot world (`search_world`):
  * drones fly at fixed heights (1.2 / 1.35 / 1.5 m above the start floor) with
    cameras pitched down 15 deg; they plan on the air layer of the observed map,
    so furniture below cruise height does not block them;
  * drones have a battery measured in seconds of flight; it drains whenever a
    drone is airborne, and an empty battery grounds it for the rest of the
    mission. They recharge only by docking on the dog, which is a moving base;
  * only the dog may declare the target found (switchable: `declare_roles`).

Drone collision: there is no flight physics. Drones only move along routes
through known, air-free, inflated cells, and an on-board forward depth check
stops any step towards an obstacle nearer than STOP_M. Crashes are not modelled
(future work). The dog keeps navmesh `try_step` collision.

Nothing a policy may read comes from the navmesh, annotations or geodesics;
those serve episode generation and scoring only.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import quaternion  # noqa: F401

from search_world import (DETECT_RANGE, HFOV, TARGETS, SUCCESS_DISTANCE, Tracker, geodesic,
                          horizontal_distance, scene_ref, success_viewpoints, target_instances)

CONTROL_DT = 0.1
MISSION_S = 300.0
DRONE_ENDURANCE_S = 150.0
SWAP_S = 15.0
DOCK_RADIUS = 0.8
DRONE_ALTS = (1.2, 1.35, 1.5)
DRONE_PITCH = -math.radians(15.0)
DRONE_SPEED = 2.0
DOG_SPEED = 1.0
DOG_CAMERA = 0.5
LOOK_PITCH_MAX = math.radians(35.0)   # head / body tilt of the legged robot for a look (looks only)
RESOLUTION = 640          # 384 px left small objects 'too blurry' for the second opinion (HM3D val)
HALF_EXTENT_CAP = 0.8
CLOSE_VIEW_SLACK_M = 0.75
DETECT_EVERY = 10          # 1.0 s; halves detector wall time
OPENING_SCAN_S = 3.5       # the opening full turn (drones 2.1 s, dog 3.2 s): detector every 0.3 s, else a drone
                           # sees only 2-3 headings of it and misses a target beside the start (qyAac8rV8Zk:9)
MAP_EVERY = 3
STOP_M = 0.35
CAREFUL_S = 1.5         # a drone in a tight spot follows its path cell by cell this long
DRONES = ('d0', 'd1', 'd2')
DOG = 'dog0'


def team_specs(resolution=RESOLUTION):
    from team_sim import RobotSpec
    specs = [RobotSpec(name=n, kind='aerial', sensor_height=0.0, hfov=HFOV,
                       resolution=(resolution, resolution), altitude=a, max_speed=DRONE_SPEED)
             for n, a in zip(DRONES, DRONE_ALTS)]
    specs.append(RobotSpec(name=DOG, kind='ground', sensor_height=DOG_CAMERA, hfov=HFOV,
                           resolution=(resolution, resolution), max_speed=DOG_SPEED,
                           camera_forward=0.3))
    return specs


def make_episode(scene_id, seed, min_geodesic=3.0):
    """Frozen episode: start pose for the dog (drones take off beside it) and a
    target category. Navmesh use here is task setup only."""
    import random
    from team_sim import TeamSim
    ref = scene_ref(scene_id)
    sim = TeamSim(ref.scene_id, ref.dataset_config, team=team_specs(), control_dt=CONTROL_DT)
    try:
        sim.build_navmeshes()
        pf = sim.pathfinder
        island = sim.main_island('ground')
        rng = random.Random(f'r28:{scene_id}:{seed}')     # the seed string fixes the released episodes
        pf.seed(rng.randrange(1 << 30))
        for _ in range(60):
            start = np.asarray(pf.get_random_navigable_point(), dtype=np.float32)
            if np.any(np.isnan(start)) or int(pf.get_island(start)) != island:
                continue
            floor_y = float(start[1])
            eligible = {}
            for cat in TARGETS:
                best = math.inf
                for inst in target_instances(sim, cat, floor_y):
                    for vp in success_viewpoints(sim, inst['obb'], floor_y, island):
                        best = min(best, geodesic(sim, start, vp))
                if math.isfinite(best) and best >= min_geodesic:
                    eligible[cat] = best
            if not eligible:
                continue
            target = rng.choice(sorted(eligible))
            return {'scene': scene_id, 'seed': seed, 'target': target, 'floor_y': floor_y,
                    'start': [float(v) for v in start], 'start_yaw': rng.uniform(-math.pi, math.pi),
                    'shortest_geodesic_m': round(eligible[target], 3),
                    'eligible_targets': {k: round(v, 3) for k, v in eligible.items()}}
        return None
    finally:
        sim.close()


class Body:
    """Kinematics, battery and dock state of one robot. Pose is kept here, not
    read back from habitat, because a pitched drone body makes habitat's
    rotation-vector yaw unreliable."""

    def __init__(self, world, spec, x, z, yaw):
        self.world, self.name, self.kind = world, spec.name, spec.kind
        self.aerial = spec.kind == 'aerial'
        self.alt = spec.altitude if self.aerial else 0.0
        self.speed = spec.max_speed
        self.x, self.z, self.yaw = float(x), float(z), float(yaw)
        self.y = world.floor_y + self.alt if self.aerial else world.floor_y
        self.battery = DRONE_ENDURANCE_S if self.aerial else math.inf
        self.state = 'airborne' if self.aerial else 'ground'   # airborne | docked | down | ground
        self.dock_left = 0.0
        self.travelled = 0.0
        self.flight_s = 0.0
        self.swaps = 0
        self.goal, self.path, self.replan_at, self.status = None, [], -1.0, 'idle'
        self.progress_ref = None
        self.face_target = None

    @property
    def layer(self):
        return 'air' if self.aerial else 'ground'

    def place(self):
        import habitat_sim
        sim = self.world.sim
        if self.aerial:
            pos = np.array([self.x, self.y, self.z], dtype=np.float32)
            rot = quaternion.from_rotation_vector([0.0, self.yaw, 0.0]) * \
                quaternion.from_rotation_vector([DRONE_PITCH, 0.0, 0.0])
            state = habitat_sim.AgentState()
            state.position, state.rotation = pos, rot
            sim.sim.get_agent(sim.index[self.name]).set_state(state)
        else:
            p = sim.set_pose(self.name, np.array([self.x, self.world.floor_y, self.z], np.float32),
                             yaw=self.yaw)
            self.x, self.y, self.z = float(p[0]), float(p[1]), float(p[2])

    # ------------------------------------------------------------ goals
    def set_goal(self, xz, t):
        if xz is None:
            self.goal, self.path, self.status = None, [], 'idle'
            return
        # A blocked body re-plans even toward the same goal: otherwise a dog stalled once stayed
        # 'blocked' for the whole mission, every re-issued verify walking 0 m (HM3D val run, 28 of 99).
        if self.goal is None or math.hypot(xz[0] - self.goal[0], xz[1] - self.goal[1]) > 0.3 or \
                self.status == 'blocked':
            self.goal = (float(xz[0]), float(xz[1]))
            self.stalls = 0                              # a new goal gets its full three tries
            self.path, self.replan_at, self.status = [], -1.0, 'moving'
            self.progress_ref = (t, self.x, self.z)

    def _plan(self, t):
        m = self.world.map
        self.replan_at = t + 1.0
        if self.aerial:
            # A drone inside an obstacle's inflation (its camera let it drift there) first flies
            # back to the nearest flyable cell: planning from inside pointed it at the obstacle every
            # time and it never moved again (an earlier HM3D val run: a drone stuck in 134 of 313 episodes).
            esc = self._escape_cell()
            if esc is not None:
                self.path = [esc]
                return
        field = m.distance_field(self.x, self.z, self.layer)
        gi, gj = m.cell(*self.goal)
        self.path = m.path_to(field, gi, gj)
        if not self.path:
            self.status = 'blocked'
            self.hold = 'no path'

    def _escape_cell(self, r=10):
        """The nearest flyable cell (world xz) when the drone's own cell is not flyable, else None."""
        m = self.world.map
        air = m._derived()['air']
        si, sj = m.cell(self.x, self.z)
        if not m.inside(si, sj) or air[si, sj]:
            return None
        i0, j0 = max(si - r, 0), max(sj - r, 0)
        win = air[i0:si + r + 1, j0:sj + r + 1]
        ii, jj = np.nonzero(win)
        if ii.size == 0:
            return None
        k = int(np.argmin((ii + i0 - si) ** 2 + (jj + j0 - sj) ** 2))
        return tuple(float(v) for v in m.world(ii[k] + i0, jj[k] + j0))

    def forward_depth(self):
        """5th-percentile depth of the centre of the robot's own depth image (inf if none)."""
        frames = self.world.frames
        if frames is None:
            return math.inf
        d = np.asarray(frames[self.world.sim.index[self.name]][f'{self.name}_depth'])
        h, w = d.shape[:2]
        patch = d[int(h * 0.30):int(h * 0.55), int(w * 0.40):int(w * 0.60)]
        patch = patch[np.isfinite(patch) & (patch > 0.02)]
        return math.inf if patch.size == 0 else float(np.percentile(patch, 5))

    def remember_air_obstacle(self, t, dist):
        """The drone's camera stopped it at something the map does not have (a thin door leaf):
        put it in the flyable layer's obstacles and re-plan now, instead of pressing on it."""
        m = self.world.map
        if getattr(m, 'collision_air', None) is None:
            m.collision_air = np.zeros_like(m.occ_air, dtype=bool)
        fx, fz, px, pz = -math.sin(self.yaw), -math.cos(self.yaw), math.cos(self.yaw), -math.sin(self.yaw)
        for ahead in (dist, dist + 0.1):
            for side in (-0.1, 0.0, 0.1):
                i, j = m.cell(self.x + ahead * fx + side * px, self.z + ahead * fz + side * pz)
                if m.inside(i, j):
                    m.collision_air[i, j] = True
        m._cache = None
        self.path, self.replan_at = [], t
        self.world.collision_log.append((round(t, 1), self.name, round(self.x, 2), round(self.z, 2)))

    def turn_to(self, want):
        rate = (3.0 if self.aerial else 2.0) * CONTROL_DT
        err = (want - self.yaw + math.pi) % (2 * math.pi) - math.pi
        self.yaw += max(-rate, min(rate, err))
        return abs(err)

    def spin(self):
        self.yaw += (3.0 if self.aerial else 2.0) * CONTROL_DT

    def forward_clear(self):
        """On-board safety: the centre of the robot's own depth image."""
        return self.forward_depth() > STOP_M

    def step(self, t):
        if self.state in ('down', 'docked'):
            return self.status
        # A step skipped (waiting for a docking drone, spinning) must not count as no progress.
        if t - getattr(self, '_last_step_t', t) > 0.15:
            self.progress_ref = None
        self._last_step_t = t
        if self.goal is None or self.status in ('idle', 'arrived', 'blocked'):
            if self.face_target is not None:
                dx, dz = self.face_target[0] - self.x, self.face_target[1] - self.z
                self.turn_to(math.atan2(-dx, -dz))
            return self.status
        # The dog stops on its goal cell (0.15 m): three right declarations missed by 0.13-0.16 m.
        if math.hypot(self.goal[0] - self.x, self.goal[1] - self.z) < (0.3 if self.aerial else 0.15):
            self.status = 'arrived'
            return self.status
        if t >= self.replan_at or not self.path:
            self._plan(t)
            if self.status == 'blocked':
                return self.status
        # Lookahead: 0.6 m for drones; the dog follows its plan closely (0.3 m), since cutting a
        # corner pinned it on a door jamb for 120 s (retest CrMo8WxCyVb:7, 27 bumps at one spot).
        # In a tight spot (a stride was refused for leaving the flyable layer) a drone follows its
        # path cell by cell and aligned first, for CAREFUL_S: aiming 0.6 m ahead with a 0.6 rad
        # heading slack cut every door frame and re-planned the same path in place (4ok3usBNeis:5).
        careful = self.aerial and t < getattr(self, 'careful_until', -1.0)
        ahead = (0.12 if careful else 0.6) if self.aerial else 0.3
        while len(self.path) > 1 and math.hypot(self.path[0][0] - self.x, self.path[0][1] - self.z) < ahead:
            self.path.pop(0)
        wx, wz = self.path[0]
        dx, dz = wx - self.x, wz - self.z
        want = math.atan2(-dx, -dz)
        if self.turn_to(want) > ((0.2 if careful else 0.6) if self.aerial else 0.35):
            self.hold = 'turning'
        else:
            stride = min(self.speed * CONTROL_DT, math.hypot(dx, dz))
            nx = self.x - math.sin(self.yaw) * stride
            nz = self.z - math.cos(self.yaw) * stride
            if self.aerial:
                m = self.world.map
                air = m._derived()['air']
                here, there = m.cell(self.x, self.z), m.cell(nx, nz)
                inside = lambda c: m.inside(*c) and bool(air[c])
                ahead = self.forward_depth()
                if ahead <= STOP_M:
                    self.hold = f'camera {ahead:.2f} m'
                    if t - getattr(self, '_air_mark_t', -math.inf) > 0.5:
                        self._air_mark_t = t
                        self.remember_air_obstacle(t, max(ahead, 0.1))
                elif inside(there) or not inside(here):
                    # never fly out of the flyable layer (cutting a corner at 0.6 m lookahead
                    # drifted drones into door frames); from outside it, only the way back
                    self.travelled += math.hypot(nx - self.x, nz - self.z)
                    self.x, self.z = nx, nz
                    self.hold = None
                else:
                    self.hold = 'would leave the flyable layer'
                    if careful:
                        self.path, self.replan_at = [], t
                    self.careful_until = t + CAREFUL_S
            else:
                sim = self.world.sim
                here = np.array([self.x, self.y, self.z], np.float32)
                stepped = np.asarray(sim.pathfinder.try_step(here, np.array([nx, self.y, nz], np.float32)))
                moved = 0.0 if np.any(np.isnan(stepped)) else math.hypot(stepped[0] - self.x, stepped[2] - self.z)
                if moved > 0.0:
                    self.travelled += moved
                    self.x, self.y, self.z = float(stepped[0]), float(stepped[1]), float(stepped[2])
        ref = self.progress_ref
        togo = math.hypot(self.goal[0] - self.x, self.goal[1] - self.z)
        if not self.aerial:
            # The dog: progress means getting 0.3 m closer to the goal (sliding along a wall
            # reset the stall count and a dog took five stalls to give up, Nfvxx8J5NCo:14).
            if ref is None or len(ref) < 4 or togo < ref[3] - 0.3:
                self.progress_ref = (t, self.x, self.z, togo)
                self.stalls = 0
                ref = None
        elif ref is None or math.hypot(self.x - ref[1], self.z - ref[2]) > 0.3:
            self.progress_ref = (t, self.x, self.z)
            self.stalls = 0
            ref = None
        if ref is not None and t - ref[0] > 4.0:
            if math.hypot(self.goal[0] - self.x, self.goal[1] - self.z) < 0.3:
                self.status = 'arrived'              # pressed against something next to its goal
                return self.status
            if self.aerial:
                self.status = 'blocked'               # drones: give the goal up now, the allocator
                return self.status                    # moves them on (3 stalls idled one for 150 s)
            # The dog: mark what it is pressing on, re-plan around it; give up after three tries.
            self.stalls = getattr(self, 'stalls', 0) + 1
            self.remember_collision(t)
            self.path, self.replan_at, self.progress_ref = [], t, (t, self.x, self.z, togo)
            if self.stalls >= 3:
                self.status = 'blocked'
        return self.status

    def remember_collision(self, t):
        """The ground robot pushed into something its map shows as free (door jambs, mesh the
        depth band missed): mark a small patch just ahead as occupied and re-plan now, so the
        planner stops sending it into the same spot."""
        m = self.world.map
        if getattr(m, 'collision', None) is None:
            m.collision = np.zeros_like(m.occ_hits, dtype=bool)
        fx, fz, px, pz = -math.sin(self.yaw), -math.cos(self.yaw), math.cos(self.yaw), -math.sin(self.yaw)
        for ahead in (0.3, 0.4):
            for side in (-0.1, 0.0, 0.1):
                i, j = m.cell(self.x + ahead * fx + side * px, self.z + ahead * fz + side * pz)
                if m.inside(i, j):
                    m.collision[i, j] = True
        m._cache = None
        self.replan_at = t
        self.world.collision_log.append((round(t, 1), self.name, round(self.x, 2), round(self.z, 2)))

    # ------------------------------------------------------------ battery
    def tick_battery(self, dog):
        if not self.aerial or self.state == 'down':
            return None
        if self.state == 'docked':
            self.dock_left -= CONTROL_DT
            self.x, self.z = dog.x, dog.z            # rides on the dog
            if self.dock_left <= 0:
                self.state, self.battery, self.swaps = 'airborne', DRONE_ENDURANCE_S, self.swaps + 1
                return 'launched'
            return None
        self.battery -= CONTROL_DT
        self.flight_s += CONTROL_DT
        if self.battery <= 0:
            self.state, self.status, self.goal = 'down', 'down', None
            return 'battery_out'
        return None

    def try_dock(self, dog, radius=DOCK_RADIUS):
        if self.aerial and self.state == 'airborne' and \
                math.hypot(self.x - dog.x, self.z - dog.z) <= radius:
            self.state, self.dock_left = 'docked', SWAP_S
            self.goal, self.path, self.status = None, [], 'idle'
            return True
        return False


def object_cells(depth, box, eye, rotation, hfov_deg, world_map, floor_y, stride=4, band=0.15):
    """Map cells under a detection's own pixels: box pixels within `band` m of the box's median
    depth, back-projected (habitat optical-axis depth, camera looking down -Z)."""
    d = np.asarray(depth)
    h, w = d.shape[:2]
    x0, y0, x1, y1 = (max(0, min(w, box[0])), max(0, min(h, box[1])), max(0, min(w, box[2])), max(0, min(h, box[3])))
    ys, xs = np.mgrid[y0:max(y1, y0 + 1):stride, x0:max(x1, x0 + 1):stride]
    z = d[np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)]
    ok = np.isfinite(z) & (z > 0.05)
    if not ok.any():
        return set()
    from perception import foreground_depth
    ok &= np.abs(z - foreground_depth(z[ok])) < band      # the object's pixels, not the wall behind
    fx = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    cam = np.stack([(xs - w / 2.0) / fx * z, -(ys - h / 2.0) / fx * z, -z], axis=-1)[ok]
    pts = np.asarray(eye) + cam @ np.asarray(rotation).T
    pts = pts[(pts[:, 1] > floor_y + 0.12) & (pts[:, 1] < floor_y + 2.0)]   # not the floor in front of it
    i, j = world_map.cell(pts[:, 0], pts[:, 2])
    return set(zip(np.asarray(i).tolist(), np.asarray(j).tolist()))


def sight_line(world_map, a, b, skip_end=2):
    """No occupied cell on the straight line between cells a and b (the last `skip_end` cells,
    which touch the object's unseen parts, are not checked)."""
    from observed_map import OCCUPIED
    occ = world_map._derived()['state'] == OCCUPIED
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
    ii = np.rint(np.linspace(a[0], b[0], n)).astype(int)[:max(1, n - skip_end)]
    jj = np.rint(np.linspace(a[1], b[1], n)).astype(int)[:max(1, n - skip_end)]
    return not bool(occ[ii, jj].any())


OWN_VIEW_S = 1.5     # the legged robot's own detection this recent measures reach (not the accumulated cells)
CELL_VIEW_M = 3.0     # object cells only from views this close (far boxes bleed onto walls)
CELL_MIN_HITS = 2     # and only cells seen at least twice


OBJECTNAV_SUCCESS_M = 0.1          # habitat ObjectNav: success when distance to a goal view point < 0.1
OBJECTNAV_SUCCESS_RELAXED_M = 0.12  # the relaxed threshold we also report


def objectnav_goals(ref, _cache={}):
    """Goal objects (with view points) of one HM3D ObjectNav episode: ref = {'file', 'key'}."""
    import gzip
    import json
    # Episode files record the goal file relative to the ObjectNav v2 root (SCOUT_OBJECTNAV_ROOT, the directory
    # holding val/ and train/ of objectnav_hm3d_v2).
    f = os.path.join(os.environ.get('SCOUT_OBJECTNAV_ROOT', ''), ref['file'])
    if f not in _cache:
        _cache.clear()
        _cache[f] = json.load(gzip.open(f))['goals_by_category']
    return _cache[f][ref['key']]


class TeamWorld:
    def __init__(self, episode, detector=None, resolution=RESOLUTION, declare_roles=('dog',),
                 batch_detection=False, localizer=None, localize_mode='box'):
        # Batching was measured and rejected: batch of 2 ran at 141 ms/frame against
        # 142 ms single (the GPU is already saturated by one 800-px pass), only 35
        # of 40 frames reproduced exactly, and a batch of 4 does not fit beside vLLM.
        from team_sim import TeamSim
        from aerial_map import AerialMap
        self.episode = episode
        self.floor_y = float(episode['floor_y'])
        ref = scene_ref(episode['scene'])
        self.specs = team_specs(resolution)
        self.sim = TeamSim(ref.scene_id, ref.dataset_config, team=self.specs, control_dt=CONTROL_DT)
        self.sim.build_navmeshes()
        s = episode['start']
        dog_start = np.asarray(self.sim.pathfinder.snap_point(np.asarray(s, np.float32)))
        self.bodies = {}
        for k, spec in enumerate(self.specs):
            if spec.kind == 'aerial':
                ang = 2 * math.pi * k / len(DRONES)
                x, z = float(dog_start[0] + 0.4 * math.cos(ang)), float(dog_start[2] + 0.4 * math.sin(ang))
                yaw = episode['start_yaw'] + ang
            else:
                x, z, yaw = float(dog_start[0]), float(dog_start[2]), episode['start_yaw']
            self.bodies[spec.name] = Body(self, spec, x, z, yaw)
        for b in self.bodies.values():
            b.place()
        import team_sim
        from aerial_map import GROUND_HEADROOM_MARGIN
        self.map = AerialMap((s[0], s[2]), floor_y=self.floor_y,
                             ground_top=team_sim.GROUND_NAV_HEIGHT + GROUND_HEADROOM_MARGIN)
        self.tracker = Tracker()
        self.obs = {}             # track id -> [(range m, position, width m)] per detection
        self.detector = detector
        self.frames = None
        self.tick = 0
        self.trails = {n: [] for n in self.bodies}
        self.declare_roles = tuple(declare_roles)
        self._instances = target_instances(self.sim, episode['target'], self.floor_y)
        self._viewpoints = None
        if episode.get('objectnav'):      # HM3D ObjectNav: the benchmark's own goal objects, any floor
            goals = objectnav_goals(episode['objectnav'])
            ids = {g['object_id'] for g in goals}
            self._instances = [{'instance_id': int(o.semantic_id), 'obb': o.obb, 'center': list(o.obb.center)}
                               for o in self.sim.sim.semantic_scene.objects
                               if o is not None and int(o.semantic_id) in ids]
            self._viewpoints = np.array([v['agent_state']['position'] for g in goals for v in g['view_points']],
                                        dtype=np.float32)
            if not self._instances or not len(self._viewpoints):
                raise RuntimeError(f"no goal objects/view points for {episode['episode_id']}")
        self._categories = {int(o.semantic_id): o.category.name()
                            for o in self.sim.sim.semantic_scene.objects
                            if o is not None and o.category is not None}
        self.detector_calls = 0
        self.dense_until = -1.0      # a look-around: detector every 0.3 s until then
        self.collision_log = []
        self.batch_detection = batch_detection
        # Optional SAM mask localisation of target-label detections (mask_localize).
        # 'box' keeps the box-median placement but still logs both estimates
        # when a localizer is given, so the two can be compared on the same views.
        self.localizer, self.localize_mode = localizer, localize_mode
        self.loc_eval = []

    # The board and search_world helpers expect these names.
    @property
    def names(self):
        return list(self.bodies)

    @property
    def movers(self):
        return self.bodies

    @property
    def dog(self):
        return self.bodies[DOG]

    def pose(self, name):
        b = self.bodies[name]
        return b.x, b.z, b.yaw

    def _close(self, track):
        """The views taken from nearest the object: depth error grows with
        range and far boxes mix in background, so a track's all-view median can
        sit a metre off a large object (a development run: the dog declared from 1.9 m of
        a real sofa it believed it was beside)."""
        obs = self.obs.get(track.track_id)
        if not obs:
            return None
        dmin = min(o[0] for o in obs)
        return [o for o in obs if o[0] <= dmin + CLOSE_VIEW_SLACK_M]

    def locate(self, track):
        """Object position from its closest views (x, y, z)."""
        close = self._close(track)
        if not close:
            return np.asarray(track.position)
        return np.median(np.stack([o[1] for o in close]), axis=0)

    def half_extent(self, track):
        """Half the object's estimated width, capped: a sofa's centre can sit a
        metre inside it, out of reach of any standing point (a development run: the dog
        stood 1.07 m from a real sofa's centre for 45 s, unable to declare)."""
        close = self._close(track)
        return min(0.5 * float(np.median([o[2] for o in close])), HALF_EXTENT_CAP) if close else 0.0

    def reach(self, name, track, seen_here=False):
        """Estimated distance from a robot to the object's near surface: to the nearest of the
        object's own cells when the dog has them, else centre estimate minus half width.
        seen_here: the dog's own camera named the target from where it stands, which is the
        sight line; the map's 2-D one fails for a TV on a cabinet (p53SfW6mjZe:12: five
        "tv_monitor" looks at P 0.8, never in declare range, 60 s between two standoffs)."""
        b = self.bodies[name]
        ov = getattr(track, 'own_view', None)
        if name == DOG and ov and getattr(self, 'now', -math.inf) - ov['t'] <= OWN_VIEW_S:
            return float(math.hypot(ov['xz'][0] - b.x, ov['xz'][1] - b.z))
        cells = getattr(track, 'cells', None)
        if name == DOG and cells:
            # Nearest object cell the dog can see: one on the far side of a wall does not count.
            ij = np.asarray(list(cells))
            cx, cz = self.map.world(ij[:, 0], ij[:, 1])
            d = np.hypot(np.asarray(cx) - b.x, np.asarray(cz) - b.z)
            if seen_here:
                return float(d.min())
            here = self.map.cell(b.x, b.z)
            for k in np.argsort(d)[:20]:
                if sight_line(self.map, here, tuple(ij[k])):
                    return float(d[k])
            # None in sight: finite (it goes into the Jev evidence as JSON; inf made every dog
            # decision fail, 35 of 85 full-run episodes) and beyond any declare range.
            return float(d.min()) + 1.0
        p = self.locate(track)
        d = math.hypot(float(p[0]) - b.x, float(p[2]) - b.z)
        return max(0.0, d - self.half_extent(track))

    def aim_point(self, name, track):
        """Where the legged robot looks for a candidate: the point its own detector last put it at, when it has
        one (the fused estimate can sit on a wall behind the object), else the fused estimate."""
        loc = np.asarray(self.locate(track), dtype=np.float64)
        ov = getattr(track, 'own_view', None)
        if name == DOG and ov:
            return np.array([ov['xz'][0], loc[1], ov['xz'][1]])
        return loc

    def look_crop(self, name, track):
        """The robot's own current image of where the candidate should be: its
        position projected into this robot's camera and cut out with context.
        Used when the dog stands at a candidate but the detector has not fired on
        it from there (development runs: 44 of 59 verify trips to real targets ended with the
        dog in range and no crop to ask about). None if it is behind the camera or
        off the image. A point in front but above or below the frame makes the legged robot tilt its head
        (LOOK_PITCH_MAX) for this one frame: from its standoff a 0.5 m camera cannot see a wall-mounted TV, every
        look came back None and the dog stood by a real TV until its trip timed out (MP3D aayBHfsNo7d:v2s23)."""
        from perception import crop_box
        k = next(i for i, sp in enumerate(self.specs) if sp.name == name)
        rgb = self.frames[k][f'{name}_rgb'] if self.frames is not None else None
        if rgb is None:
            return None
        aim = np.asarray(self.aim_point(name, track), dtype=np.float64)
        box = self._look_box(name, k, aim, rgb.shape, track)
        if box is None and name == DOG:
            eye, _ = self.sim.camera_transform(name)
            pitch = math.atan2(aim[1] - eye[1], math.hypot(aim[0] - eye[0], aim[2] - eye[2]))
            pitch = max(-LOOK_PITCH_MAX, min(LOOK_PITCH_MAX, pitch))
            if abs(pitch) > math.radians(5.0):
                b = self.bodies[name]
                self.sim.set_pose(name, np.array([b.x, self.floor_y, b.z], np.float32), yaw=b.yaw, pitch=pitch)
                rgb = self.sim.sim.get_sensor_observations([k])[k][f'{name}_rgb']
                box = self._look_box(name, k, aim, rgb.shape, track)
                b.place()                      # level again
        return None if box is None else crop_box(rgb, box)

    def _look_box(self, name, k, aim, shape, track):
        h, w = shape[:2]
        eye, R = self.sim.camera_transform(name)
        cam = np.asarray(R).T @ (aim - eye)
        if cam[2] > -0.1:                      # the camera looks down -Z
            return None
        fx = (w / 2.0) / math.tan(math.radians(self.specs[k].hfov) / 2.0)
        cx, cy = w / 2.0 + fx * cam[0] / -cam[2], h / 2.0 - fx * cam[1] / -cam[2]
        if not (0 <= cx < w and 0 <= cy < h):
            return None
        half = max(40.0, fx * max(self.half_extent(track), 0.25) / -cam[2])
        return (int(max(0, cx - half)), int(max(0, cy - half)), int(min(w, cx + half)), int(min(h, cy + half)))

    def can_declare(self, name):
        kind = 'dog' if name == DOG else 'drone'
        return kind in self.declare_roles

    def sense(self, t):
        self.now = t
        """Render, map, and detect. All robots detect on the same tick, so the
        detector can run once for the whole team (see batched_detection)."""
        from perception import crop_box, ground_detections
        for b in self.bodies.values():
            if b.state != 'down':
                b.place()
        self.frames = self.sim.observe()
        touched, lookers = [], []
        for k, spec in enumerate(self.specs):
            b = self.bodies[spec.name]
            if b.state == 'down':
                continue
            if not b.aerial:
                self.map.stand(b.x, b.z, t)
            if self.tick % 5 == 0:
                self.trails[b.name].append((round(b.x, 2), round(b.z, 2)))
            if b.state == 'docked':
                continue
            if (self.tick + k) % MAP_EVERY == 0:
                eye, R = self.sim.camera_transform(spec.name)
                self.map.integrate(self.frames[k][f'{spec.name}_depth'], eye, R, spec.hfov, t)
            if self.detector is not None and self.tick % (3 if t < max(self.dense_until, OPENING_SCAN_S) else DETECT_EVERY) == 0:
                lookers.append((k, spec, b))
        if lookers:
            rgbs = [self.frames[k][f'{spec.name}_rgb'] for k, spec, _ in lookers]
            if self.batch_detection:
                from batched_detection import detect_batch
                all_dets = detect_batch(self.detector, rgbs)
            else:
                all_dets = [self.detector.detect(r) for r in rgbs]
            self.detector_calls += len(lookers)
            for (k, spec, b), rgb, raw in zip(lookers, rgbs, all_dets):
                depth = self.frames[k][f'{spec.name}_depth']
                sem = self.frames[k].get(f'{spec.name}_semantic')
                dets = ground_detections(raw, depth, self.sim, spec.name, floor_y=self.floor_y)
                widths, evals = {}, {}
                if self.localizer is not None:
                    tgt = [d for d in dets if d.position is not None and d.label == self.episode['target']]
                    if tgt:
                        from mask_localize import MaskLocalizer
                        eye, R = self.sim.camera_transform(spec.name)
                        for d, m in zip(tgt, self.localizer.masks(rgb, [d.box for d in tgt])):
                            r = MaskLocalizer.place(m, depth, eye, R, spec.hfov, d.box)
                            if r is None:
                                continue
                            pos, wid, _ = r
                            evals[id(d)] = {'t': round(t, 2), 'robot': spec.name, 'range': round(d.distance, 2),
                                            'box': [round(float(v), 3) for v in d.position],
                                            'mask': [round(float(v), 3) for v in pos], 'mask_width': round(wid, 2),
                                            # evaluation only, never read by any policy
                                            'gt_box_m': self._gt_distance(d.position),
                                            'gt_mask_m': self._gt_distance(pos)}
                            if self.localize_mode == 'mask':
                                d.position = pos
                                d.distance = float(np.linalg.norm(np.asarray(pos) - np.asarray(eye)))
                                widths[id(d)] = wid
                for det in dets:
                    if det.position is None or det.distance > DETECT_RANGE or det.label == '?':
                        continue
                    h = float(det.position[1]) - self.floor_y
                    if not (-0.3 <= h <= 2.4):
                        continue
                    dx, dz = det.position[0] - b.x, det.position[2] - b.z
                    truth = []
                    if sem is not None:
                        x0, y0, x1, y1 = det.box
                        ids, counts = np.unique(np.asarray(sem)[y0:y1, x0:x1], return_counts=True)
                        truth = [int(i_) for i_, c_ in zip(ids, counts)
                                 if int(i_) in self._categories and c_ >= 64 and
                                 self._categories[int(i_)] == det.label]
                    tr = self.tracker.add(det.label, np.asarray(det.position), det.score, spec.name, t,
                                          det.distance, math.atan2(dx, dz), crop_box(rgb, det.box), truth)
                    if os.environ.get('SCOUT_CROPS') and sem is not None:     # evaluation only, never read by a policy
                        x0, y0, x1, y1 = det.box
                        ids, counts = np.unique(np.asarray(sem)[y0:y1, x0:x1], return_counts=True)
                        top = [(self._categories[int(i_)], int(c_)) for i_, c_ in sorted(zip(ids, counts), key=lambda x: -x[1])
                               if int(i_) in self._categories][:3]
                        if not hasattr(tr, 'crop_gt'):
                            tr.crop_gt = {}
                        tr.crop_gt[spec.name] = (t, top)
                    if spec.name == DOG and det.label == self.episode['target'] and det.position is not None:
                        # What the legged robot measures right now (foreground depth of its own box): the
                        # accumulated cells can hold one stray near cell, and reach took the nearest -- dogs
                        # 1.7-2.3 m from the object "in range" (an earlier HM3D val run: 61 declarations on real tracks).
                        tr.own_view = {'t': t, 'xz': (float(det.position[0]), float(det.position[2]))}
                        if os.environ.get('SCOUT_DIAG') and sem is not None:      # evaluation only, never read by a policy
                            x0, y0, x1, y1 = det.box
                            dp, sp = np.asarray(depth)[y0:y1, x0:x1], np.asarray(sem)[y0:y1, x0:x1]
                            own = np.isin(sp, [i['instance_id'] for i in self._instances]) & (dp > 0.05)
                            ids, cnt = np.unique(sp, return_counts=True)
                            top = [(self._categories.get(int(i_), '?'), int(c_)) for i_, c_ in
                                   sorted(zip(ids, cnt), key=lambda x: -x[1])[:4]]
                            with open(os.environ['SCOUT_DIAG'], 'a') as fh:
                                fh.write(json.dumps({'t': round(t, 2), 'track': tr.track_id, 'box': list(map(int, det.box)),
                                    'est_depth': round(float(det.distance), 2),
                                    'target_px_frac': round(float(own.mean()), 3),
                                    'target_px_depth': round(float(np.median(dp[own])), 2) if own.any() else None,
                                    'est_to_gt_m': self._gt_distance(det.position), 'box_classes': top}) + '\n')
                    if det.label == self.episode['target'] and det.distance <= CELL_VIEW_M:
                        # The cells the object's own pixels land on, not just one median point: the
                        # dog stands and stops by them (as SemExp/VLFM do with the goal's map cells).
                        eye_, R_ = self.sim.camera_transform(spec.name)
                        if not hasattr(tr, 'cell_hits'):
                            tr.cell_hits, tr.cells = {}, set()
                        for c in object_cells(depth, det.box, eye_, R_, spec.hfov, self.map, self.floor_y):
                            tr.cell_hits[c] = tr.cell_hits.get(c, 0) + 1
                            if tr.cell_hits[c] >= CELL_MIN_HITS:
                                tr.cells.add(c)
                    if id(det) in evals:
                        self.loc_eval.append({**evals[id(det)], 'track': tr.track_id, 'truth': truth})
                    # Physical width from the mask if placed by it, else from the box
                    # (pixels / focal length x range).
                    fw = depth.shape[1]
                    fx = (fw / 2.0) / math.tan(math.radians(spec.hfov) / 2.0)
                    width = widths.get(id(det), (det.box[2] - det.box[0]) / fx * det.distance)
                    self.obs.setdefault(tr.track_id, []).append(
                        (float(det.distance), np.asarray(det.position, dtype=np.float64), width))
                    touched.append(tr)
        self.tick += 1
        return touched

    # ------------------------------------------------------------ evaluation only
    def _gt_distance(self, p):
        best = min((horizontal_distance(i['obb'], float(p[0]), float(p[2])) for i in self._instances), default=math.inf)
        return round(best, 3) if math.isfinite(best) else None

    def score_declaration(self, name):
        b = self.bodies[name]
        best = min((horizontal_distance(i['obb'], b.x, b.z) for i in self._instances), default=math.inf)
        out = {'success': best <= SUCCESS_DISTANCE, 'distance_to_nearest_target_m': round(best, 3),
               'declared_at': [round(b.x, 3), round(b.z, 3)], 'dtg_from': 'floor_y'}
        if self._viewpoints is not None:
            d = self.viewpoint_distance(name)
            out['objectnav_dtg_m'] = d
            out['objectnav_success'] = d is not None and d < OBJECTNAV_SUCCESS_M          # habitat: < 0.1
            out['objectnav_success_012'] = d is not None and d <= OBJECTNAV_SUCCESS_RELAXED_M
        return out

    def viewpoint_distance(self, name):
        """Habitat ObjectNav DistanceToGoal: geodesic from the robot to the nearest goal view point."""
        import habitat_sim
        # On the scene's own navmesh, as habitat's DistanceToGoal is: ours is recomputed for our
        # robots, and a dog standing 0.04 m from a view point measured 0.155 m on it (Nfvxx8J5NCo:4).
        if getattr(self, '_official_pf', None) is None:
            self._official_pf = habitat_sim.PathFinder()
            nav = scene_ref(self.episode['scene']).scene_id.replace('.glb', '.navmesh')
            if not self._official_pf.load_nav_mesh(nav):
                raise RuntimeError(f'official navmesh did not load: {nav}')
        b = self.bodies[name]
        # At the benchmark agent's base height (floor_y: the episode start and the view points are
        # stored there), not b.y: try_step on OUR navmesh leaves the dog ~0.15 m higher, and that
        # height went into the geodesic -- dogs 0.03 m from a view point scored 0.15 m (an earlier HM3D val run).
        start = np.array([b.x, self.floor_y, b.z], dtype=np.float32)
        self.dtg_start_navigable = bool(self._official_pf.is_navigable(start))
        path = habitat_sim.MultiGoalShortestPath()
        path.requested_start = start
        path.requested_ends = self._viewpoints
        ok = self._official_pf.find_path(path)
        return round(float(path.geodesic_distance), 3) if ok and math.isfinite(path.geodesic_distance) else None

    def track_is_real(self, track):
        return bool(track.truth)

    def close(self):
        self.sim.close()

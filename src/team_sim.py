"""Heterogeneous robot team in habitat-sim: N aerial agents + M ground agents.

Aerial agents get an unconstrained 6-DOF pose (no navmesh filtering), ground
agents are snapped to the navmesh so walls and furniture actually block them.
Locomotion itself is abstracted away on both sides -- an upstream controller is
assumed to realise the commanded pose. The research layer above this cares about
who looks at what, not about rotor dynamics or gait.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import quaternion  # noqa: F401  (registers np.quaternion, pulled in by habitat_sim)

import habitat_sim


def bounded_yaw(current, desired, max_delta):
    """One turn step from `current` toward `desired`, at most `max_delta` rad, wrapped to [-pi, pi)."""
    if not all(math.isfinite(x) for x in (current, desired, max_delta)) or max_delta <= 0:
        raise ValueError('Invalid turn step')
    error = (desired - current + math.pi) % (2 * math.pi) - math.pi
    delta = max(-max_delta, min(max_delta, error))
    return (current + delta + math.pi) % (2 * math.pi) - math.pi


@dataclass
class RobotSpec:
    """One robot in the team."""

    name: str
    kind: str  # "aerial" | "ground"
    sensor_height: float  # camera height above the robot origin, metres
    resolution: tuple[int, int] = (256, 256)
    hfov: float = 90.0
    # hover height above the flying navmesh, metres (EMOS uses 0.8 on MP3D)
    altitude: float = 0.8
    # ground speed, metres per second. Multiplied by the caller's control
    # timestep to get a stride, so the number means the same thing whatever the
    # tick rate is. It used to be metres per *tick*, which at SIM_DT=0.1 had the
    # drones flying at 12 m/s and the quadruped at 4 m/s -- three times what a
    # Go2 can do. That error flattered us: brain latency is charged against this
    # clock, so robots that move too fast make a reply arriving 2 s late look
    # more costly than it really is, which is the effect we are claiming.
    max_speed: float = 1.0
    # push the camera clear of the robot's own body, metres forward
    camera_forward: float = 0.0


def default_team(n_aerial: int = 3, n_ground: int = 1) -> list[RobotSpec]:
    """Three drones plus one quadruped -- the asymmetry is the point.

    Drones cruise high, move fast and see a lot from a distance. The quadruped
    is slow, stuck on the floor and blocked by furniture, but its low camera
    gets close enough to settle what the drones can only guess at.
    """
    team = [
        RobotSpec(
            name=f"drone{i}",
            kind="aerial",
            sensor_height=0.0,
            altitude=0.8 + 0.15 * i,  # stagger so views are not co-planar
            max_speed=2.0,   # m/s; a cautious indoor cruise
            camera_forward=0.35,  # clear of the scaled airframe
        )
        for i in range(n_aerial)
    ]
    team += [
        RobotSpec(
            name=f"dog{i}",
            kind="ground",
            sensor_height=0.35,  # quadruped head height
            hfov=70.0,
            max_speed=1.0,   # m/s; a Go2 tops out near 1.5
            camera_forward=0.32,  # at the Go2's head, not inside its torso
        )
        for i in range(n_ground)
    ]
    return team


def _sensor_specs(robot: RobotSpec) -> list[habitat_sim.CameraSensorSpec]:
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = f"{robot.name}_rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = list(robot.resolution)
    rgb.position = [0.0, robot.sensor_height, -robot.camera_forward]
    rgb.hfov = robot.hfov
    rgb.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    depth = habitat_sim.CameraSensorSpec()
    depth.uuid = f"{robot.name}_depth"
    depth.sensor_type = habitat_sim.SensorType.DEPTH
    depth.resolution = list(robot.resolution)
    depth.position = [0.0, robot.sensor_height, -robot.camera_forward]
    depth.hfov = robot.hfov
    depth.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # Semantic frames are how we measure what a robot can actually see. Counting
    # an object's pixels is exact, handles occlusion for free, and avoids
    # re-deriving habitat's camera conventions by hand.
    sem = habitat_sim.CameraSensorSpec()
    sem.uuid = f"{robot.name}_semantic"
    sem.sensor_type = habitat_sim.SensorType.SEMANTIC
    sem.resolution = list(robot.resolution)
    sem.position = [0.0, robot.sensor_height, -robot.camera_forward]
    sem.hfov = robot.hfov
    sem.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    return [rgb, depth, sem]


GROUND_NAV_RADIUS = 0.20   # the dog's collision radius on the ground navmesh
GROUND_NAV_HEIGHT = 0.50   # headroom the dog needs on the ground navmesh

class TeamSim:
    """A habitat-sim scene holding a heterogeneous team, addressed by name."""

    def __init__(
        self,
        scene_id: str,
        scene_dataset_config: str | None = None,
        team: list[RobotSpec] | None = None,
        enable_physics: bool = True,
        gpu_device_id: int = 0,
        load_semantic_mesh: bool = True,
        control_dt: float = 0.1,
    ):
        self.team = team if team is not None else default_team()
        self.scene_id = scene_id
        # Seconds of simulated time per move_towards() call. Speeds and turn
        # rates are per second and get multiplied by this, so the motion is the
        # same whatever tick rate a caller chooses.
        self.control_dt = float(control_dt)
        self.index = {r.name: i for i, r in enumerate(self.team)}

        backend = habitat_sim.SimulatorConfiguration()
        backend.scene_id = scene_id
        if scene_dataset_config:
            backend.scene_dataset_config_file = scene_dataset_config
        backend.enable_physics = enable_physics
        backend.gpu_device_id = gpu_device_id
        # MP3D ships per-object and per-room annotations; they are the only
        # source of real inspection targets we have, so always try to load them.
        backend.load_semantic_mesh = load_semantic_mesh

        agent_cfgs = []
        for robot in self.team:
            cfg = habitat_sim.agent.AgentConfiguration()
            cfg.sensor_specifications = _sensor_specs(robot)
            # Motion is driven through set_state(), so the discrete action space
            # stays unused. It still has to be non-empty for habitat-sim.
            cfg.action_space = {
                "stay": habitat_sim.agent.ActionSpec(
                    "move_forward", habitat_sim.agent.ActuationSpec(amount=0.0)
                )
            }
            agent_cfgs.append(cfg)

        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, agent_cfgs))
        # cached geodesic route per ground robot, consumed by move_towards()
        self._routes: dict[str, list[np.ndarray]] = {}
        self._route_goal: dict[str, np.ndarray] = {}
        self._stalls: dict[str, int] = {}
        self._place_initial()

    # ----------------------------------------------------------------- navmeshes

    def build_navmeshes(self) -> dict[str, float]:
        """One navmesh per robot kind -- this is where the asymmetry comes from.

        The ground mesh is a normal walkable surface. The aerial mesh is built
        with a zero-radius, any-slope agent, which makes it a stand-in for free
        flight: it covers everything the ground mesh does and a good deal it
        does not. Having both means "cost for the dog to reach T" and "cost for
        a drone to reach T" are two exact numbers rather than one guess.
        """
        ground = habitat_sim.NavMeshSettings()
        ground.set_defaults()
        ground.agent_radius = GROUND_NAV_RADIUS
        ground.agent_height = GROUND_NAV_HEIGHT
        ground.agent_max_climb = 0.25  # a real quadruped takes stairs; do not fake the asymmetry
        ground.agent_max_slope = 45.0
        ground.include_static_objects = True

        # Flying-base settings taken from EMOS's drone_oracle_nav_action_3d.yaml
        # (agent_radius 0.0 + 0.05 padding, height 0.5, max_climb 0.5,
        # max_slope 90). A 90 degree slope limit is the load-bearing part: it
        # lets the mesh wrap vertical surfaces, so storeys stay connected
        # instead of fragmenting into one island per ledge.
        aerial = habitat_sim.NavMeshSettings()
        aerial.set_defaults()
        aerial.agent_radius = 0.0 + 0.05
        aerial.agent_height = 0.50
        aerial.agent_max_climb = 0.50
        aerial.agent_max_slope = 90.0
        aerial.include_static_objects = True

        self.sim.recompute_navmesh(self.sim.pathfinder, ground)
        self._ground_pf = self.sim.pathfinder

        self._aerial_pf = habitat_sim.PathFinder()
        if not self.sim.recompute_navmesh(self._aerial_pf, aerial):
            self._aerial_pf = self._ground_pf  # fall back rather than crash

        return {
            "ground_area": float(self._ground_pf.navigable_area),
            "aerial_area": float(self._aerial_pf.navigable_area),
            "ground_islands": int(self._ground_pf.num_islands),
            "aerial_islands": int(self._aerial_pf.num_islands),
        }

    def main_island(self, kind: str) -> int:
        """Index of the largest connected component of this kind's navmesh."""
        pf = self.pathfinder_for(kind)
        if not pf.is_loaded or pf.num_islands == 0:
            return -1
        return max(range(pf.num_islands), key=lambda i: pf.island_area(i))

    def island_of(self, kind: str, point, altitude: float = 0.0) -> int:
        """Which navmesh component is under this point.

        `altitude` must be subtracted for a hovering drone: snapping its lifted
        position finds whatever surface is nearest in 3D, often a ceiling on a
        different component, which is how two of three drones came to look
        stranded when they were in fact over the main floor.
        """
        pf = self.pathfinder_for(kind)
        p = np.asarray(point, dtype=np.float32).copy()
        p[1] -= altitude
        snapped = np.asarray(pf.snap_point(p), dtype=np.float32)
        if np.any(np.isnan(snapped)):
            return -1
        return int(pf.get_island(snapped))

    def island_of_robot(self, name: str) -> int:
        robot = self.team[self.index[name]]
        pos, _ = self.get_pose(name)
        alt = robot.altitude if robot.kind == "aerial" else 0.0
        return self.island_of(robot.kind, pos, altitude=alt)

    def on_main_island(self, kind: str, point, altitude: float = 0.0) -> bool:
        return self.island_of(kind, point, altitude) == self.main_island(kind)

    def spawn_on_main_island(self, name: str, near=None, seed: int = 0) -> np.ndarray:
        """Place a robot somewhere it can actually get out of.

        Each kind has its own navmesh with its own components, so a single
        shared spawn point strands somebody: dropping the whole team on one
        ground anchor put the drones on a minor aerial island with 2 of 31 tour
        goals reachable, and the quadruped on a minor ground island.
        """
        robot = self.team[self.index[name]]
        pf = self.pathfinder_for(robot.kind)
        island = self.main_island(robot.kind)
        pf.seed(seed)

        best = None
        for _ in range(500):
            p = np.asarray(pf.get_random_navigable_point(), dtype=np.float32)
            if np.any(np.isnan(p)) or int(pf.get_island(p)) != island:
                continue
            if near is None:
                best = p
                break
            d = float(np.linalg.norm((p - np.asarray(near, dtype=np.float32))[[0, 2]]))
            if best is None or d < best[0]:
                best = (d, p)
        if best is None:
            return self.get_pose(name)[0]
        point = best if near is None else best[1]
        return self.set_pose(name, point)

    def pathfinder_for(self, kind: str):
        if kind == "aerial" and getattr(self, "_aerial_pf", None) is not None:
            return self._aerial_pf
        return self.sim.pathfinder

    def navmesh_pos(self, name: str) -> np.ndarray:
        """This robot's position projected back onto its own navmesh.

        Aerial robots hover `altitude` above the flying mesh, so their agent
        position is off it. Snapping that lifted point directly finds whatever
        surface is nearest in 3D -- often a ceiling or a ledge on a different
        island -- which made every reach cost come back as unreachable.
        """
        robot = self.team[self.index[name]]
        pos, _ = self.get_pose(name)
        pos = np.asarray(pos, dtype=np.float32).copy()
        if robot.kind == "aerial":
            pos[1] -= robot.altitude
        pf = self.pathfinder_for(robot.kind)
        if pf.is_loaded:
            snapped = np.asarray(pf.snap_point(pos), dtype=np.float32)
            if not np.any(np.isnan(snapped)):
                return snapped
        return pos

    def reach_cost(self, name: str, target) -> float:
        """Geodesic distance for this robot to reach `target`, or inf.

        Computed, never estimated -- the brain is given this number so it does
        not have to reason about geometry it cannot see.
        """
        robot = self.team[self.index[name]]
        pf = self.pathfinder_for(robot.kind)
        if not pf.is_loaded:
            return float("inf")
        path = habitat_sim.ShortestPath()
        path.requested_start = self.navmesh_pos(name)
        path.requested_end = np.asarray(pf.snap_point(np.asarray(target, dtype=np.float32)),
                                        dtype=np.float32)
        if np.any(np.isnan(path.requested_start)) or np.any(np.isnan(path.requested_end)):
            return float("inf")
        if not pf.find_path(path) or not np.isfinite(path.geodesic_distance):
            return float("inf")
        return float(path.geodesic_distance)

    # ---------------------------------------------------------------- placement

    @property
    def pathfinder(self):
        return self.sim.pathfinder

    def _navmesh_ready(self) -> bool:
        return self.pathfinder is not None and self.pathfinder.is_loaded

    def _place_initial(self) -> None:
        """Drop everyone near a navigable seed point, drones lifted off the floor."""
        if self._navmesh_ready():
            seed = self.pathfinder.get_random_navigable_point()
        else:
            seed = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        for i, robot in enumerate(self.team):
            offset = np.array([0.6 * math.cos(i * 1.6), 0.0, 0.6 * math.sin(i * 1.6)])
            self.set_pose(robot.name, np.asarray(seed) + offset, yaw=i * 1.2)

    # ------------------------------------------------------------------- control

    def set_pose(self, name: str, position, yaw: float = 0.0, pitch: float = 0.0,
                 lifted: bool = False) -> np.ndarray:
        """Teleport a robot. Ground robots are snapped down onto the navmesh.

        Only for initial placement and resets -- snap_point finds the *nearest*
        navigable point, which can sit on the far side of a thin wall. Use
        move_towards() for anything that has to respect geometry.

        `lifted` says whether `position` already carries a drone's hover offset.
        It has to be said rather than guessed. Snapping a hovering position
        directly and then lifting again adds the offset a second time whenever
        the nearest navmesh point is not the one underneath -- which is common,
        since a drone at ceiling height is often nearer a mesh across the room
        than the floor below it. Two of three drones climbed out of their own
        navmesh island that way in the first 300 ticks and then had no route to
        anywhere for the rest of the episode.
        """
        robot = self.team[self.index[name]]
        pos = np.asarray(position, dtype=np.float32).copy()
        aerial = robot.kind == "aerial"
        if aerial and lifted:
            pos[1] -= robot.altitude

        pf = self.pathfinder_for(robot.kind)
        if pf.is_loaded:
            snapped = np.asarray(pf.snap_point(pos), dtype=np.float32)
            if not np.any(np.isnan(snapped)):
                pos = snapped

        if aerial:
            # Navigate on the flying navmesh but hover above it, the way EMOS
            # offsets its drone by `base_offset = (0, -height, 0)`. Forcing an
            # absolute altitude instead decouples the drone from the mesh its
            # own routes are planned on, and it ends up skimming the floor.
            pos = pos.copy()
            pos[1] += robot.altitude

        # habitat is y-up; yaw about y, pitch about the local x axis
        rot = quaternion.from_rotation_vector([0.0, yaw, 0.0])
        if pitch:
            rot = rot * quaternion.from_rotation_vector([pitch, 0.0, 0.0])

        state = habitat_sim.AgentState()
        state.position = pos
        state.rotation = rot
        self.sim.get_agent(self.index[name]).set_state(state)
        return pos

    def get_pose(self, name: str) -> tuple[np.ndarray, float]:
        state = self.sim.get_agent(self.index[name]).get_state()
        pos = np.asarray(state.position, dtype=np.float32)
        yaw = quaternion.as_rotation_vector(state.rotation)[1]
        return pos, float(yaw)

    def camera_transform(self, name: str, sensor: str = "depth"):
        """Actual camera-to-world extrinsics, including mount offset and pitch."""
        state = self.sim.get_agent(self.index[name]).get_state()
        camera = state.sensor_states[f"{name}_{sensor}"]
        return (np.asarray(camera.position, dtype=np.float64).copy(),
                quaternion.as_rotation_matrix(camera.rotation))

    def turn_towards(self, name: str, desired_yaw: float) -> float:
        """One bounded turn at the current position; never re-snap translation."""
        agent = self.sim.get_agent(self.index[name])
        state = agent.get_state()
        _, current = self.get_pose(name)
        yaw = bounded_yaw(current, desired_yaw, self.TURN_RATE * self.control_dt)
        state.rotation = quaternion.from_rotation_vector([0., yaw, 0.])
        agent.set_state(state)
        return yaw

    # Stop this far short of whatever a drone is about to fly into.
    AERIAL_CLEARANCE = 0.25

    # Stop this far short of whatever a drone is about to fly into.
    AERIAL_XY_TOLERANCE = 0.35

    # --- motion -------------------------------------------------------------
    # Waypoint controller modelled on EMOS's OracleNavDiffBaseAction: take the
    # next point off habitat's own geodesic path, turn in place until roughly
    # facing it, then drive forward. Separating turning from translation is what
    # keeps robots on the route instead of cutting corners through walls.

    # radians per second. Was 0.45 per tick, i.e. 4.5 rad/s -- 258 degrees a
    # second, which no quadruped turns at.
    TURN_RATE = 2.0
    DIST_THRESH = 0.30   # close enough to a waypoint
    TURN_THRESH = 0.30   # radians; beyond this, turn before moving
    REPLAN_DIST = 0.50   # goal moved further than this -> replan

    @staticmethod
    def _wrap(a: float) -> float:
        return float((a + np.pi) % (2 * np.pi) - np.pi)

    @staticmethod
    def _yaw_towards(delta: np.ndarray) -> float:
        """Yaw that points habitat's -Z forward axis along `delta`."""
        return float(np.arctan2(-delta[0], -delta[2]))

    def forward_vector(self, yaw: float) -> np.ndarray:
        return np.array([-np.sin(yaw), 0.0, -np.cos(yaw)], dtype=np.float32)

    def plan_route(self, name: str, target) -> list[np.ndarray]:
        """Geodesic waypoints on this robot's own navmesh. Empty if unreachable."""
        robot = self.team[self.index[name]]
        pf = self.pathfinder_for(robot.kind)
        if not pf.is_loaded:
            return []
        path = habitat_sim.ShortestPath()
        path.requested_start = self.navmesh_pos(name)
        path.requested_end = np.asarray(
            pf.snap_point(np.asarray(target, dtype=np.float32)), dtype=np.float32)
        if np.any(np.isnan(path.requested_start)) or np.any(np.isnan(path.requested_end)):
            self._routes[name] = []
            return []
        if not pf.find_path(path):
            self._routes[name] = []
            return []
        pts = [np.asarray(p, dtype=np.float32) for p in path.points]
        self._routes[name] = pts[1:] if len(pts) > 1 else pts
        return self._routes[name]

    def move_towards(self, name: str, target, yaw: float | None = None) -> np.ndarray:
        """One control tick towards `target`: turn if badly aimed, else advance."""
        robot = self.team[self.index[name]]
        pos, cur_yaw = self.get_pose(name)
        target = np.asarray(target, dtype=np.float32)

        prev_goal = self._route_goal.get(name)
        if prev_goal is None or float(np.linalg.norm(prev_goal - target)) > self.REPLAN_DIST:
            self.plan_route(name, target)
            self._route_goal[name] = target.copy()

        route = self._routes.get(name) or []
        while route:
            flat = (route[0] - pos).copy()
            flat[1] = 0.0
            if float(np.linalg.norm(flat)) > self.DIST_THRESH:
                break
            route.pop(0)
        if not route:
            self._routes[name] = []
            self._route_goal.pop(name, None)
            return pos  # arrived, or no route exists at all

        wp = route[0]
        delta = (wp - pos).copy()
        delta[1] = 0.0
        dist = float(np.linalg.norm(delta))
        want_yaw = self._yaw_towards(delta)
        yaw_err = self._wrap(want_yaw - cur_yaw)

        if abs(yaw_err) > self.TURN_THRESH:
            # Turn on the spot. Costing a tick for this is the point: it is what
            # makes the quadruped genuinely slower to redirect than a drone.
            rate = self.TURN_RATE * self.control_dt
            step = float(np.clip(yaw_err, -rate, rate))
            return self.set_pose(name, pos, yaw=self._wrap(cur_yaw + step),
                                 lifted=True)

        stride = min(robot.max_speed * self.control_dt, dist)
        goal = pos + self.forward_vector(cur_yaw) * stride

        if robot.kind == "ground" and self._navmesh_ready():
            stepped = self.pathfinder.try_step(pos, goal)
            if not np.any(np.isnan(stepped)):
                goal = np.asarray(stepped, dtype=np.float32)

        return self.set_pose(name, goal, yaw=want_yaw, lifted=True)



    def observe(self) -> dict[int, dict]:
        """One RGB-D frame per robot, keyed by agent index."""
        return self.sim.get_sensor_observations(list(range(len(self.team))))

    def close(self) -> None:
        self.sim.close()

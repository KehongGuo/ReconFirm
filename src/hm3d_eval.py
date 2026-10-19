"""HM3D ObjectNav evaluation entry (run.py with `dataset: hm3d`).

Same runner, arguments and method as team_run.py. It patches the core at run time for the benchmark protocol
only: the naming standard (category_standard: detector aliases, what counts as the target, category boundaries in
the prompt, track truth from the goal objects) and the body and camera of the benchmark agent (with the standoff its
camera height needs). Whether the legged robot also explores is a run option (--dog-explores; no in
configs/hm3d_val.yaml: single-floor homes). Scoring is the core's (official < 0.1 m to a goal view point, plus the relaxed 0.12 m variant).

  python src/hm3d_eval.py --episodes data/episodes/team_episodes_hm3d_val.json --episode-id <id> <team_run args>
"""
import json
import os
import sys
from pathlib import Path

import aerial_map
import category_standard as cs
import verifier_prompt
import search_agents
import team_run
import team_sim
import team_world


def install(target):
    team_run.SOURCES = tuple(team_run.SOURCES) + ('hm3d_eval.py', 'category_standard.py', 'team_sim.py',
                                                  os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'hm3d_category_standard.json'))
    search_agents.PHRASE[target] = cs.describe(target)

    # --- the agent's camera: at 0.88 m (+-39.5 deg) it loses a 0.4 m toilet's top closer than ~0.6 m, so its
    # standoff ends at 0.6 m (the core's 0.85 m is for the 0.5 m legged-robot camera); declaring range is the core's
    team_run.Runner.standoff.__defaults__ = (0.35, 0.6, 1.0, None)      # (lo, hi, fallback, avoid)

    # --- robot size (drones ~15 cm wide; the dog as the benchmark agent moves: the scene navmeshes
    # are built with radius 0.10 m, and its camera sits on the body centre at 0.88 m)
    aerial_map.AIR_RADIUS = 0.10
    aerial_map.GROUND_RADIUS = 0.15        # plan inflation: body + a thin margin; bumps are remembered
    team_sim.GROUND_NAV_RADIUS = 0.10
    team_sim.GROUND_NAV_HEIGHT = 0.88
    team_world.DOG_CAMERA = 0.88
    base_specs = team_world.team_specs

    def team_specs(resolution=team_world.RESOLUTION):
        specs = base_specs(resolution)
        for sp in specs:
            if sp.kind == 'ground':
                sp.camera_forward = 0.0    # 0.3 m forward put the camera inside a toilet (q3zU7Yy5E5s:10)
        return specs
    team_world.team_specs = team_specs

    # --- naming standard: detector aliases, second-opinion answers, goal-object truth
    aliases = cs.detector_aliases(target)
    base_load = team_run.load_detector

    def load_detector():
        det = base_load()
        det.set_vocabulary(aliases + det.vocabulary)
        detect = det.detect

        def detect_std(rgb):
            out = detect(rgb)
            for d in out:
                if d.label in aliases:
                    d.label = target
            return out
        det.detect = detect_std
        return det
    team_run.load_detector = load_detector

    base_verify = team_run.verify

    def verify(server, crop, vocabulary):
        ans = base_verify(server, crop, list(vocabulary) + aliases + ['picture'])
        if ans['category']:
            ans['category'] = cs.canonical(ans['category'], target)
        return ans
    team_run.verify = verify

    base_init = team_world.TeamWorld.__init__

    def init(self, episode, *a, **k):
        base_init(self, episode, *a, **k)
        for i in self._instances:                   # the benchmark's goal objects are the target
            self._categories[i['instance_id']] = episode['target']
    team_world.TeamWorld.__init__ = init

    # category boundaries in this environment, after the core prompt
    verifier_prompt.PROMPT = (verifier_prompt.PROMPT + '\nCategory boundaries in this environment '
                                 '(other categories, not the one named): ' + cs.boundaries(target) + '.')



def main():
    argv = sys.argv[1:]
    doc = json.loads(Path(argv[argv.index('--episodes') + 1]).read_text())
    eid = argv[argv.index('--episode-id') + 1]
    ep = next(e for e in doc['episodes'] if e['episode_id'] == eid)
    install(ep['target'])
    team_run.main()


if __name__ == '__main__':
    main()

"""Who decides what: team allocation, local semantic choices, and their services.

Two layers, each with two interchangeable implementations, so a 2x2 design can
attribute any effect to one layer:

  allocation  geometric   greedy frontier assignment, instant, no model
              brain       the local VLM reads the tactical board (image + JSON)
                          and assigns every robot a frontier; slow, asynchronous
  local       rule        fixed thresholds, instant
              jev         TypeSafe Jev at semantic decision points

The local layer owns two decisions: what to do about a possible target
(continue / approach / get a second opinion / declare / reject), and which
frontier to take when a robot needs one before the allocator has spoken.

Everything a policy reads comes from the tactical board or the tracker, i.e.
from what the team has observed. Nothing here imports evaluation truth.

Thresholds and wordings below were fixed before any comparison was run.
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import time

import numpy as np

# The scheduler/verifier VLM: an OpenAI-compatible server (vLLM) serving Qwen3-VL-8B under SCOUT_VLM_MODEL.
VLM_ENDPOINT = os.environ.get('SCOUT_VLM_URL', 'http://localhost:8010') + '/v1/chat/completions'
VLM_MODEL = os.environ.get('SCOUT_VLM_MODEL', 'qwen3vl')

PHRASE = {'tv_monitor': 'television or computer monitor', 'plant': 'potted plant'}
DECLARE_RANGE = 1.0        # a robot may only declare within this of the candidate
RULE_APPROACH_SCORE = 0.35 # rule policy: detector score that earns a look
JEV_CONSIDER_SCORE = 0.25  # jev policy: anything the detector reports at all
MAX_CHECKS = 3             # second opinions per candidate
DECLARE_GATE = 0.5         # jev: calibrated P(target) required to declare
REJECT_GATE = 0.3          # jev: P(target) below which a reject is honoured


def phrase(target):
    return PHRASE.get(target, target.replace('_', ' '))


# ---------------------------------------------------------------- the VLM server
class VLMServer:
    """One local vLLM process serving one request at a time.

    Calls run synchronously in wall time while the simulation is paused; the
    measured latency is then charged on the mission clock by the runner, and a
    request issued while another is still 'in flight' in simulated time queues
    behind it, as it would on a single-sequence server.
    """

    def __init__(self, endpoint=VLM_ENDPOINT):
        import requests
        self.endpoint = endpoint
        self.session = requests.Session()
        self.busy_until = 0.0
        self.calls = 0

    def chat(self, images, text, max_tokens=200, model=None):
        content = [{'type': 'image_url',
                    'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(b).decode()}}
                   for b in images]
        content.append({'type': 'text', 'text': text})
        payload = {'model': model or VLM_MODEL, 'messages': [{'role': 'user', 'content': content}],
                   'max_tokens': max_tokens, 'temperature': 0.0, 'seed': 0}
        t0 = time.perf_counter()
        r = self.session.post(self.endpoint, json=payload, timeout=180)
        r.raise_for_status()
        latency = time.perf_counter() - t0
        self.calls += 1
        body = r.json()
        return body['choices'][0]['message']['content'], latency, body.get('usage', {})

    def schedule(self, now, latency):
        """Simulated release time for a request issued at `now`."""
        start = max(now, self.busy_until)
        self.busy_until = start + latency
        return self.busy_until


# ---------------------------------------------------------------- brain
BRAIN_PROMPT = """You are the coordinator of a team of {n} ground robots searching an unknown building for a {target}.

The image is the team's shared map, built only from what the robots have seen so far. Dark grey is not yet seen. White is seen free floor. Black is walls and obstacles. Light blue is where robots have walked. Coloured patches labelled F1, F2, ... are frontiers: openings into unseen space. Coloured dots numbered 0, 1, 2 are robots r0, r1, r2, with a line showing where each faces. Brown text marks objects the robots' detector has seen. Red circles labelled T.. are possible {target}s.

The JSON repeats the map in numbers. path_m is the walking distance through already-seen space from each robot to each frontier.

{board}

Assign every robot one frontier to explore next. Think about which kind of room a {target} is usually in, and what the objects seen near each frontier suggest about the room beyond it. Spread the robots over different frontiers unless there are fewer frontiers than robots. When nothing suggests a better direction, prefer the shorter path. A robot that is investigating a candidate keeps its job, so you may leave it out.

Answer with JSON only, in exactly this form:
{{"assignments": [{{"robot": "<robot id>", "frontier": "<frontier id>", "why": "<short reason>"}}]}}
Angle brackets are placeholders: fill them with ids from the board above, never copy them."""


def parse_brain(text):
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        return None
    try:
        doc = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = []
    for a in doc.get('assignments', []) if isinstance(doc, dict) else []:
        if isinstance(a, dict) and isinstance(a.get('robot'), str) and isinstance(a.get('frontier'), str):
            out.append({'robot': a['robot'].strip(), 'frontier': a['frontier'].strip().upper(),
                        'why': str(a.get('why', ''))[:200]})
    return out


class Brain:
    def __init__(self, server, n_robots):
        self.server, self.n = server, n_robots

    def plan(self, board_json, board_png, target):
        text = BRAIN_PROMPT.format(n=self.n, target=phrase(target),
                                   board=json.dumps(board_json, separators=(',', ':')))
        raw, latency, usage = self.server.chat([board_png], text, max_tokens=220)
        return {'raw': raw, 'assignments': parse_brain(raw), 'latency_s': latency,
                'usage': usage, 'prompt_chars': len(text)}


# ---------------------------------------------------------------- second opinion
def verify(server, crop_jpeg, vocabulary):
    """The verifier's question: open, closed-set, never told the detector's label."""
    import re
    from verifier_prompt import build_prompt, parse
    vocabulary = list(vocabulary) + ([] if 'picture' in vocabulary else ['picture'])   # depicted, not real
    prompt = build_prompt(vocabulary)
    # SCOUT_VERIFIER_MODEL: an evolved verifier, a LoRA served beside the base (vLLM --lora-modules); the scheduler
    # keeps the base model
    raw, latency, _ = server.chat([crop_jpeg], prompt, max_tokens=120, model=os.environ.get('SCOUT_VERIFIER_MODEL'))
    ans = parse(raw, vocabulary)
    return {'category': ans['category'], 'clear': ans['clear'], 'prompt': prompt, 'vocabulary': vocabulary,
            'description': ans['description'], 'raw': raw, 'latency_s': latency,
            'reflection': bool(re.search(r'REFLECTION\W*\s*yes', raw or '', re.I))}


# ---------------------------------------------------------------- geometric allocation
def geometric_pick(name, frontiers, tasks):
    """Greedy utility: path cost, a bonus for wide exits, and a penalty for
    exits a teammate is already heading to. The classic coordinated-frontier
    heuristic, deterministic and instant."""
    best, best_score = None, math.inf
    teammate_goals = [tk['goal'] for n, tk in tasks.items() if n != name and tk and tk.get('goal')]
    for f in frontiers.values():
        d = f['cost'].get(name, math.inf)
        if not math.isfinite(d) or name not in f['goal']:
            continue
        width = f['size'] * 0.1
        score = d - 0.5 * min(width, 4.0)
        if any(tk and tk.get('fid') == f['id'] for n, tk in tasks.items() if n != name):
            score += 4.0
        gx, gz = f['goal'][name]
        if any(math.hypot(gx - a, gz - b) < 3.0 for a, b in teammate_goals):
            score += 2.0
        if score < best_score:
            best, best_score = f, score
    return best


# ---------------------------------------------------------------- local: rule
class RuleLocal:
    kind = 'rule'

    def choose_frontier(self, name, frontiers, tasks, world, target, now):
        f = geometric_pick(name, frontiers, tasks)
        return (f['id'] if f else None), 0.0, {'policy': 'rule'}

    def candidate(self, name, track, ctx, now):
        """Fixed thresholds. Returns (action, latency_s, record)."""
        checks = track.vlm          # second opinions are shared by the whole team
        clear = [c for c in checks if c['clear']]
        if any(c['category'] == ctx['target'] for c in clear) and ctx['distance_m'] <= DECLARE_RANGE:
            return 'declare_found', 0.0, {'policy': 'rule'}
        if clear and all(c['category'] != ctx['target'] for c in clear):
            return 'reject', 0.0, {'policy': 'rule'}
        if len(checks) >= MAX_CHECKS and not clear:
            return 'reject', 0.0, {'policy': 'rule'}
        if ctx['handled_by'] not in (None, name):
            return 'continue_task', 0.0, {'policy': 'rule'}
        if max(track.scores) < RULE_APPROACH_SCORE:
            return 'continue_task', 0.0, {'policy': 'rule'}
        if ctx['fresh_view'] and ctx['distance_m'] <= 2.5 and len(checks) < MAX_CHECKS and \
                not ctx['check_pending']:
            return 'second_opinion', 0.0, {'policy': 'rule'}
        return 'approach', 0.0, {'policy': 'rule'}


# ---------------------------------------------------------------- local: jev
DETECTOR_MODEL = {
    'what_it_is': 'one open-vocabulary object detector, the same model on every robot',
    'reliability': ('Its category names are often wrong, especially for boxes on walls, '
                    'in clutter or far away. A single detection is weak evidence.'),
    'agreement': ('Because every robot runs the same detector, several robots reporting the '
                  'same label is one model answering repeatedly, not independent confirmation.'),
    'score': "max_score and mean_score are the detector's own uncalibrated confidence.",
    'reflections': ('Detections inside mirrors are reflections: their position is behind the mirror and wrong. '
                    'Objects shown in paintings, photos, posters or screens are pictures, not the object.'),
}
SECOND_OPINION_MODEL = (
    'Entries in `candidate.vlm_checks` come from a different model, a vision-language model '
    'shown the robot\'s image without being told what the detector said. It names the object '
    'in the centre from a fixed list (or "none"), says whether the image was clear, and '
    'describes it in one sentence. This is the only evidence independent of the detector.')


class JevLocal:
    kind = 'jev'

    def __init__(self, client):
        self.client = client

    def _call(self, state, questions):
        t0 = time.perf_counter()
        answers = self.client.judge(state, questions)
        return answers, time.perf_counter() - t0

    def choose_frontier(self, name, frontiers, tasks, world, target, now):
        from tactical_board import nearby_labels
        x, z, yaw = world.pose(name)
        opts = sorted((f for f in frontiers.values() if math.isfinite(f['cost'].get(name, math.inf))
                       and name in f['goal']), key=lambda f: f['cost'][name])[:5]
        if not opts:
            return None, 0.0, {'policy': 'jev', 'note': 'no reachable frontier'}
        if len(opts) == 1:
            return opts[0]['id'], 0.0, {'policy': 'jev', 'note': 'single option, not asked'}
        criteria = {}
        for f in opts:
            cx, cz = f['centroid']
            rel = math.atan2(-(cx - x), -(cz - z)) - yaw
            rel = (rel + math.pi) % (2 * math.pi) - math.pi
            side = ('ahead' if abs(rel) < 0.6 else 'behind' if abs(rel) > 2.5
                    else 'to the left' if rel > 0 else 'to the right')
            mates = [n for n, tk in tasks.items() if n != name and tk and tk.get('fid') == f['id']]
            near = nearby_labels(world, cx, cz)
            criteria[f['id']] = (
                f"{f['cost'][name]:.1f} m walk, {f['size'] * 0.1:.1f} m wide opening, {side}. "
                f"Objects seen near it: {', '.join(l.replace('_', ' ') for l in near) or 'none yet'}. "
                + (f"Teammate {', '.join(mates)} is already heading there." if mates
                   else 'No teammate is heading there.'))
        state = {
            'mission': (f'Three robots are searching an unknown building for a {phrase(target)}. '
                        f'{max(0, round(ctx_left(world, now)))} s remain.'),
            'this_robot': {'id': name, 'needs': 'a new unexplored opening to head for'},
            'teammates': [{'id': n, 'task': (tk['fid'] if tk and tk.get('kind') == 'frontier'
                                              else ('candidate ' + tk['track_id']) if tk else 'none')}
                          for n, tk in tasks.items() if n != name],
        }
        questions = {'frontier': {
            'type': 'choice',
            'instructions': (f'Which opening should `this_robot` explore next to find a '
                             f'{phrase(target)} soonest? Prefer an opening whose nearby objects '
                             f'suggest the kind of room a {phrase(target)} is usually in. Avoid '
                             f'openings a teammate is already heading to. When nothing else '
                             f'distinguishes them, prefer the shorter walk.'),
            'criteria': criteria}}
        answers, latency = self._call(state, questions)
        a = answers['frontier']
        return a['choice'], latency, {'policy': 'jev', 'probabilities': a['probabilities'],
                                      'confidence': a['confidence'],
                                      'usage': getattr(self.client, 'last_usage', None)}

    def candidate(self, name, track, ctx, now):
        target = ctx['target']
        feasible = {
            'continue_task': ('Keep doing `this_robot.task` and leave this candidate for now. '
                              'Right when the evidence is weak, or a teammate is closer or already '
                              'handling it.'),
            'approach': (f'Walk to within 1 m of the candidate and face it, about '
                         f'{ctx["path_m"]:.1f} m. Right when it is plausible and nobody else is '
                         f'handling it.'),
            'reject': (f'Mark the candidate as not a {phrase(target)}, for the whole team. Right '
                       f'when the independent second opinion clearly names something else.'),
        }
        if ctx['fresh_view'] and ctx['distance_m'] <= 4.0 and len(track.vlm) < MAX_CHECKS \
                and not ctx['check_pending']:
            feasible['second_opinion'] = (
                'Send this robot\'s current image of the candidate to the independent vision-'
                'language model. The robot keeps moving while it answers. Right when '
                '`candidate.vlm_checks` is missing or unclear and the current view is close.')
        if ctx['distance_m'] <= DECLARE_RANGE:
            feasible['declare_found'] = (
                f'Declare the {phrase(target)} found here. This ends the mission, and fails it if '
                f'the candidate is not really a {phrase(target)}.')
        state = {
            'mission': (f'Three robots are searching an unknown building for a {phrase(target)}. '
                        f'A declaration ends the mission: it succeeds only if the declaring robot '
                        f'is really within 1 m of a {phrase(target)}, and a wrong declaration '
                        f'fails the whole mission. {max(0, round(ctx["time_left_s"]))} s remain.'),
            'detector': DETECTOR_MODEL,
            'second_opinion_source': SECOND_OPINION_MODEL,
            'candidate': {**track.summary(now),
                          'distance_from_this_robot_m': round(ctx['distance_m'], 2),
                          'in_view_of_this_robot_now': ctx['fresh_view'],
                          'objects_seen_near_it': ctx['near_labels']},
            'this_robot': {'id': name, 'task': ctx['task_desc']},
            'teammates': ctx['teammates'],
            'handled_by': ctx['handled_by'] or 'nobody',
        }
        questions = {
            'is_target': {'type': 'noul', 'instructions': (
                f'Is `candidate` really a {phrase(target)}? Weigh `candidate.vlm_checks` most, since '
                f'it is independent of the detector. Then views from different positions '
                f'(`candidate.views`, `candidate.bearing_span_rad`, `candidate.closest_view_m`). '
                f'Then whether `candidate.objects_seen_near_it` fit a room where a '
                f'{phrase(target)} is usually found. Remember `detector.reliability`.')},
            'action': {'type': 'choice', 'instructions': (
                f'What should `this_robot` do about `candidate` now, to find a real '
                f'{phrase(target)} soonest without a wrong declaration?'),
                'criteria': feasible},
        }
        answers, latency = self._call(state, questions)
        p = float(answers['is_target']['value'])
        act = answers['action']['choice']
        chosen = act
        # Code keeps the last word on the two irreversible moves, using the
        # calibrated probability rather than the argmax label.
        if act == 'declare_found' and p < DECLARE_GATE:
            act = 'second_opinion' if 'second_opinion' in feasible else 'approach'
        if act == 'reject' and p >= REJECT_GATE:
            act = 'continue_task'
        return act, latency, {'policy': 'jev', 'p_target': round(p, 4), 'jev_choice': chosen,
                              'probabilities': answers['action']['probabilities'],
                              'confidence': answers['action']['confidence'],
                              'offered': sorted(feasible),
                              'usage': getattr(self.client, 'last_usage', None)}


def ctx_left(world, now):
    from search_world import MISSION_SECONDS
    return MISSION_SECONDS - now

"""Scheduling and local decisions for three drones and one dog.

Scheduler (team level) -- typed tasks, not frontier picks:
  explore(F)   drone or dog heads for frontier F
  inspect(T)   a drone flies close to candidate T for a close view
  verify(T)    the dog walks to candidate T to confirm it (the costly resource)
  recharge     a drone flies back to the dog to swap its battery
  hold         stay put
Two schedulers share that task space: `geometric` (greedy frontiers; the dog is
dispatched to any candidate the detector scores >= 0.35, which is where "the
dog runs for nothing" comes from) and `brain` (the local VLM reads the board).

Local decisions (per robot, at semantic decision points), three
interchangeable deciders -- `rule`, `jev`, `vlm` -- placeable independently on
the drones and on the dog, which is the placement ablation of design section 3.
Code keeps the last word on irreversible moves for every decider: a declaration
or a team-wide rejection is gated on the decider's P(target) (0.5 / 0.3). For
`jev` that probability is calibrated by training; for `vlm` it is self-reported.

Battery safety is code, not policy: see `needs_recharge`.
"""
from __future__ import annotations

import json
import math
import re
import time

from search_agents import (DETECTOR_MODEL, MAX_CHECKS, RULE_APPROACH_SCORE, SECOND_OPINION_MODEL,
                           VLMServer, phrase, verify)
SECOND_OPINION_MODEL = SECOND_OPINION_MODEL + (
    ' For small objects such as plants, "blurry" or "unclear" does not mean the object is absent.')
from team_world import DOG, DOG_SPEED, DRONE_SPEED, DRONES

DECLARE_RANGE = 1.0
DECLARE_GATE = 0.5
REJECT_GATE = 0.3
DOG_REQUEST_GATE = 0.5      # drone local: P(target) to ask for the dog
RESERVE_S = 12.0            # battery margin a drone keeps on top of the flight home


# ---------------------------------------------------------------- battery rule
def flight_home_s(world, name, field_to_dog=None):
    b, dog = world.bodies[name], world.dog
    return math.hypot(b.x - dog.x, b.z - dog.z) * 1.3 / DRONE_SPEED


def needs_recharge(world, name):
    b = world.bodies[name]
    return b.aerial and b.state == 'airborne' and b.battery <= flight_home_s(world, name) + RESERVE_S


def affordable(world, name, goal):
    """Can a drone fly to `goal` and then to the dog without running dry?"""
    b, dog = world.bodies[name], world.dog
    out = math.hypot(goal[0] - b.x, goal[1] - b.z) * 1.3
    back = math.hypot(goal[0] - dog.x, goal[1] - dog.z) * 1.3
    return b.battery >= (out + back) / DRONE_SPEED + RESERVE_S


# ---------------------------------------------------------------- geometric scheduler
def pick_frontier(name, frontiers, tasks):
    best, best_score = None, math.inf
    goals = [tk['goal'] for n, tk in tasks.items() if n != name and tk and tk.get('goal')]
    for f in frontiers.values():
        d = f['cost'].get(name, math.inf)
        if not math.isfinite(d) or name not in f['goal']:
            continue
        score = d - 0.5 * min(f['size'] * 0.1, 4.0)
        if any(tk and tk.get('target') == f['id'] for n, tk in tasks.items() if n != name):
            score += 4.0
        gx, gz = f['goal'][name]
        if any(math.hypot(gx - a, gz - b) < 3.0 for a, b in goals):
            score += 2.0
        if score < best_score:
            best, best_score = f, score
    return best


# ---------------------------------------------------------------- brain
BRAIN_PROMPT = """You are the scheduler of a search team in an unknown building: three drones (d0, d1, d2) and one legged robot dog (dog0). The team is looking for a {target}. There is no map or overhead view; everything below was built from the robots' own depth cameras during this mission.

What each robot is for:
- Drones fly fast (2 m/s) at head height over furniture and see a lot, but their batteries are short (battery_s is seconds of flight left) and they can only recharge by flying back to the dog. A drone that runs out is lost for the mission. Drones cannot declare the target found.
- The dog walks slowly (1 m/s) and never runs out, but it is the only robot that may declare the target found, and only from within 1 m of it. Sending the dog to a candidate that turns out not to be a {target} wastes its time.

Task types you can assign:
- explore(F): go to frontier F (drones or dog)
- inspect(T): a drone flies close to candidate T to get a clearer view (drones only)
- verify(T): the dog walks to candidate T to confirm it (dog only)
- recharge: a drone flies back to the dog to swap batteries (drones only)
- hold: stay where you are

The image is the shared map: dark grey unseen, white seen free space, black obstacles, light blue where robots went. Coloured labelled patches F1, F2, ... are frontiers: edges of the searched area, beyond which space is unseen or was only glimpsed from more than 3 m away (too far to trust the detector). Triangles are drones, the square is the dog. Brown text marks objects the detector reported; red circles T.. are possible {target}s. The JSON repeats it in numbers; second_opinions are answers from a separate vision model shown the robot's image without being told what the detector said.

{board}

`options` lists, for each robot, the tasks that are feasible for it right now with the travel distance: every one is reachable through seen space, affordable on that drone's battery and allowed for that robot. Choose exactly one option for each robot from ITS OWN list, copying its task and target; anything else will be refused.

Assign every robot one task. Spread the drones over different frontiers when there is nothing better to do. Think about which kind of room a {target} is usually in and what the objects near each frontier suggest; if team_experience is present, it summarises what earlier missions verified. Only send the dog to verify a candidate when the evidence makes it worth the walk; use a drone to inspect when a closer look would settle it more cheaply. Send a drone to recharge before its battery_s is less than the time it needs to reach the dog.

Answer with JSON only, in this form. Angle brackets are placeholders: fill them with ids from the board above, never copy them.
{{"assignments": [{{"robot": "<robot id>", "task": "<task from its options>", "target": "<id from its options>", "why": "<short reason>"}}]}}"""

BRAIN_PROMPT_HYBRID = """You are the scheduler of a search team in an unknown building: three drones (d0, d1, d2) and one legged robot dog (dog0). The team is looking for a {target}. There is no map or overhead view; everything below was built from the robots' own depth cameras during this mission.

What each robot is for:
- Drones fly fast (2 m/s) at head height over furniture and see a lot, but their batteries are short (battery_s is seconds of flight left) and they can only recharge by flying back to the dog. A drone that runs out is lost for the mission. Drones cannot declare the target found.
- The dog walks slowly (1 m/s) and never runs out, but it is the only robot that may declare the target found, and only from within 1 m of it. Sending the dog to a candidate that turns out not to be a {target} wastes its time.

You make two kinds of decision; code does the geometry.
1. search_order: rank the frontiers (F1, F2, ...) by how likely the {target} is to be found beyond them. Use common sense about which kind of room a {target} is usually in and what the objects seen near each frontier suggest about the room beyond it (landmarks_near). Put the most promising first; you may leave out frontiers that are clearly unpromising. Code sends every robot without a task below to a frontier, weighing your order against travel time, and keeps the robots spread out.
2. assignments: only for robots that should do something other than search now. `options` lists the only tasks each robot may take right now: inspect(T) (a drone flies close to candidate T for a clearer view), verify(T) (the dog walks to candidate T to confirm it; offered only when the evidence for T is strong enough), recharge (a drone flies back to the dog). Leave a robot out to keep it searching. Tasks already under way are not interrupted.

The image is the shared map: dark grey unseen, white seen free space, black obstacles, light blue where robots went. Coloured labelled patches F1, F2, ... are frontiers: edges of the searched area. Triangles are drones, the square is the dog. Brown text marks objects the detector reported; red circles T.. are possible {target}s. second_opinions are answers from a separate vision model shown the robot's image without being told what the detector said.

{board}

Answer with JSON only, in this form. Angle brackets are placeholders: fill them with ids from the board above, never copy them.
{{"search_order": ["<frontier id>", "<frontier id>", "..."], "assignments": [{{"robot": "<robot id>", "task": "<task from its options>", "target": "<id from its options>", "why": "<short reason>"}}]}}
Use "assignments": [] when every robot should keep searching."""

TASK_TYPES = {'explore', 'inspect', 'verify', 'recharge', 'hold'}


def _reply_json(text):
    """The reply's JSON object; a reply cut off by max_tokens is closed after its last complete element
    (6.7% of scheduler replies in the HM3D val rerun ended inside a long "why" and lost the whole plan)."""
    start = text.find('{')
    if start < 0:
        return None
    text = text[start:]
    m = re.search(r'\{.*\}', text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    stack, in_str, esc, cuts = [], False, False, []
    for i, ch in enumerate(text):           # bracket depth outside strings; cut points after complete elements
        if in_str:
            esc, in_str = (not esc and ch == '\\'), in_str and (esc or ch != '"')
            continue
        if ch == '"':
            in_str = True
        elif ch in '{[':
            stack.append('}' if ch == '{' else ']')
        elif ch in '}]' and stack:
            stack.pop()
            cuts.append((i + 1, ''.join(reversed(stack))))
    for end, closers in reversed(cuts):
        try:
            return json.loads(text[:end] + closers)
        except json.JSONDecodeError:
            continue
    return None


def parse_order(text):
    """Frontier ranking from a hybrid-scheduler reply; [] if absent."""
    doc = _reply_json(text)
    order = doc.get('search_order', []) if isinstance(doc, dict) else []
    out = []
    for f in order if isinstance(order, list) else []:
        f = str(f).strip().upper().split()[0] if str(f).strip() else ''
        if re.fullmatch(r'F\d+', f) and f not in out:
            out.append(f)
    return out


def parse_plan(text):
    doc = _reply_json(text)
    if doc is None:
        return None
    out = []
    for a in (doc.get('assignments', []) if isinstance(doc, dict) else []):
        if not isinstance(a, dict):
            continue
        robot, task = str(a.get('robot', '')).strip(), str(a.get('task', '')).strip().lower()
        target = a.get('target')
        target = (str(target).strip().upper().split() or [None])[0] if target not in (None, '', 'null', 'none') else None
        if robot and task in TASK_TYPES:
            out.append({'robot': robot, 'task': task, 'target': target, 'why': str(a.get('why', ''))[:200]})
    return out


class Brain:
    def __init__(self, server, hybrid=False):
        self.server, self.hybrid = server, hybrid

    def plan(self, board_json, board_png, target):
        prompt = BRAIN_PROMPT_HYBRID if self.hybrid else BRAIN_PROMPT
        text = prompt.format(target=phrase(target), board=json.dumps(board_json, separators=(',', ':')))
        raw, latency, usage = self.server.chat([board_png], text, max_tokens=260)
        out = {'raw': raw, 'assignments': parse_plan(raw), 'latency_s': latency, 'usage': usage}
        if self.hybrid:
            out['order'] = parse_order(raw)
            if out['assignments'] is None and out['order']:
                out['assignments'] = []
        return out


# ---------------------------------------------------------------- local deciders
def _options(role, ctx, track):
    """Code-bound feasible actions for this role right now."""
    t = phrase(ctx['target'])
    opts = {'continue_task': ('Keep doing the current task and leave this candidate for now. Right when '
                              'the evidence is weak or someone else is already handling it.'),
            'reject': (f'Mark the candidate as not a {t}, for the whole team. Right when the independent '
                       f'second opinion clearly names something else.')}
    fresh = ctx['fresh_view'] and ctx['distance_m'] <= 4.0 and len(track.vlm) < MAX_CHECKS \
        and not ctx['check_pending']
    if fresh:
        opts['second_opinion'] = ('Send this robot\'s current image of the candidate to the independent '
                                  'vision model. The robot keeps moving while it answers.')
    if role == 'drone' and not ctx.get('inspected'):
        opts['inspect'] = (f'Fly close to the candidate for a clearer view, about {ctx["distance_m"]:.1f} m. '
                           f'Costs battery ({ctx["battery_s"]:.0f} s left).')
    if role == 'drone':
        if not ctx.get('dog_requested'):
            opts['request_dog'] = (f'Ask for the dog to come and confirm it. The dog walks at 1 m/s and '
                                   f'is {ctx["dog_distance_m"]:.1f} m away; a wasted trip costs the '
                                   f'team that time.')
    else:
        reach = ctx.get('reach_m', ctx['distance_m'])      # to the object's estimated near surface
        if reach > DECLARE_RANGE:
            opts['approach'] = f'Walk to within 1 m of the candidate and face it, about {ctx["path_m"]:.1f} m.'
        # The dog is the close-range verifier: it declares only after a second opinion
        # from its own camera names the target (a development run: the dog walked to a
        # wall-mounted "TV" another robot reported and declared at P = 0.75 without
        # ever having looked at it itself).
        if reach <= DECLARE_RANGE and ctx['can_declare'] and ctx.get('own_confirmation', True):
            opts['declare_found'] = (f'Declare the {t} found here. This ends the mission, and fails it if '
                                     f'the candidate is not really a {t}.')
    return opts


def _state(role, name, track, ctx, now):
    return {
        'mission': (f'Three drones and a robot dog are searching an unknown building for a '
                    f'{phrase(ctx["target"])}. Only the dog can declare it found, from within 1 m; a '
                    f'wrong declaration fails the whole mission. {max(0, round(ctx["time_left_s"]))} s remain.'),
        'detector': DETECTOR_MODEL,
        'second_opinion_source': SECOND_OPINION_MODEL,
        'candidate': {**track.summary(now), 'distance_from_this_robot_m': round(ctx['distance_m'], 2),
                      'distance_to_its_near_surface_m': round(ctx.get('reach_m', ctx['distance_m']), 2),
                      'in_view_of_this_robot_now': ctx['fresh_view'],
                      'objects_seen_near_it': ctx['near_labels']},
        'this_robot': {'id': name, 'role': role, 'task': ctx['task_desc'],
                       **({'battery_s': round(ctx['battery_s'])} if role == 'drone' else {})},
        'teammates': ctx['teammates'],
        'handled_by': ctx['handled_by'] or 'nobody',
        'dog_already_requested': bool(ctx.get('dog_requested')),
        **({'team_experience': ctx['lessons']} if ctx.get('lessons') else {}),
    }


def reject_backed(track, target):
    """A team-wide rejection is irreversible, so it needs evidence against the
    candidate: a clear second opinion naming something else, or two opinions.
    (A development run: a real TV seen once, one unclear "chair" opinion, Jev P = 0.09,
    rejected for the whole team -- the target was lost for the mission.)"""
    # "none" usually means the crop missed the object (bad framing), not that it is
    # something else: two "none" answers let the team reject two real targets.
    vlm = getattr(track, 'vlm', []) or []
    # Opinions that name another class; one naming the target is not evidence against it (two answers
    # 'bathtub', 'toilet' rejected a real toilet team-wide, XB4GS9ShBRE:9 in the HM3D val rerun).
    checks = [c for c in vlm if c['category'] not in ('none', target)]
    # ...but the dog's own clear look finding nothing there is: without it a false candidate
    # held the dog for 50 s with five 'none' answers, three of them clear (HY1NcmCgn3n:17).
    dog_clear_none = any(c['clear'] and c['category'] == 'none' and c.get('by') == 'dog0' for c in vlm)
    return any(c['clear'] for c in checks) or len(checks) >= 2 or dog_clear_none


DECLARE_GATE_UNCLEAR = 0.7


def _declare_gate(track, target):
    """If the dog's own look named the target only unclearly, ask for more.
    Development data: declarations on an unclear own view were right 10 / false 16;
    the right ones had P 0.73-0.87, the false ones mostly 0.51-0.71."""
    own = [c for c in (getattr(track, 'vlm', []) or []) if c.get('by') == 'dog0' and c['category'] == target]
    return DECLARE_GATE_UNCLEAR if own and not any(c['clear'] for c in own) else DECLARE_GATE


def _gate(act, p, opts, track=None, target=None):
    """Irreversible moves need the probability to agree, whoever proposed them.

    And once the dog stands within declaring range, whether to declare is the
    probability's call alone: in an early development run the dog stood 0.7 m
    from a real toilet with P = 0.75 and kept choosing "approach" for 60 s,
    because an action question invites caution. The model judges the evidence;
    code takes the irreversible action (TypeSafe: "judges evidence and
    attributes, not actions")."""
    if 'declare_found' in opts and p >= _declare_gate(track, target):
        return 'declare_found'
    if act == 'declare_found' and p < _declare_gate(track, target):
        act = 'second_opinion' if 'second_opinion' in opts else 'approach'
    if act == 'reject' and p >= REJECT_GATE:
        act = 'continue_task'
    if act == 'reject' and track is not None and not reject_backed(track, target):
        act = 'second_opinion' if 'second_opinion' in opts else 'continue_task'
    # A drone that has had its close look and still believes the candidate is the
    # target calls the dog: further drone views add little (a development run: P ~ 0.65 on a
    # real TV for two minutes while the drone kept choosing an inspection it had
    # already done, and the dog was never called).
    if 'request_dog' in opts and 'inspect' not in opts and 'declare_found' not in opts and \
            p >= DOG_REQUEST_GATE and act in ('continue_task', 'inspect', 'second_opinion'):
        act = 'request_dog'
    # The dog walking to a candidate is the same cost as calling it: same gate.
    if act == 'approach' and 'declare_found' not in opts and 'request_dog' not in opts and p < DOG_REQUEST_GATE:
        act = 'second_opinion' if 'second_opinion' in opts else 'continue_task'
    if act == 'request_dog' and p < DOG_REQUEST_GATE:
        act = 'second_opinion' if 'second_opinion' in opts else 'continue_task'
    # Only an offered action may leave the gate: a vetoed declare became 'approach' for a dog
    # already in range, which the runner then executed as a no-op (Nfvxx8J5NCo:14, 13 times).
    if act not in opts:
        act = 'second_opinion' if 'second_opinion' in opts else 'continue_task'
    return act


class RuleDecider:
    kind = 'rule'

    def candidate(self, role, name, track, ctx, now):
        opts = _options(role, ctx, track)
        checks = track.vlm
        clear = [c for c in checks if c['clear']]
        match = any(c['category'] == ctx['target'] for c in clear)
        rec = {'policy': 'rule', 'offered': sorted(opts)}
        if match and 'declare_found' in opts:
            return 'declare_found', 0.0, rec
        if clear and not match:
            return 'reject', 0.0, rec
        if len(checks) >= MAX_CHECKS and not clear:
            return 'reject', 0.0, rec
        if match and 'request_dog' in opts:
            return 'request_dog', 0.0, rec
        if 'second_opinion' in opts and ctx['distance_m'] <= 2.5:
            return 'second_opinion', 0.0, rec
        if max(track.scores) < RULE_APPROACH_SCORE or ctx['handled_by'] not in (None, name):
            return 'continue_task', 0.0, rec
        return ('inspect' if role == 'drone' else 'approach'), 0.0, rec

    def frontier(self, name, fronts, tasks, world, target, now):
        f = pick_frontier(name, fronts, tasks)
        return (f['id'] if f else None), 0.0, {'policy': 'rule'}


class _ModelDecider:
    """Shared by Jev and VLM: same state, same options, same gates."""

    def _frontier_state(self, name, fronts, tasks, world, target):
        from tactical_board import nearby_labels
        x, z, yaw = world.pose(name)
        opts = sorted((f for f in fronts.values() if math.isfinite(f['cost'].get(name, math.inf))
                       and name in f['goal']), key=lambda f: f['cost'][name])[:5]
        crit = {}
        for f in opts:
            cx, cz = f['centroid']
            mates = [n for n, tk in tasks.items() if n != name and tk and tk.get('target') == f['id']]
            near = nearby_labels(world, cx, cz)
            crit[f['id']] = (f"{f['cost'][name]:.1f} m away, {f['size'] * 0.1:.1f} m wide. Objects seen "
                             f"near it: {', '.join(l.replace('_', ' ') for l in near) or 'none yet'}. "
                             + (f"{', '.join(mates)} already heading there." if mates else
                                'Nobody is heading there.'))
        return opts, crit


class JevDecider(_ModelDecider):
    kind = 'jev'

    def __init__(self, client):
        self.client = client

    def candidate(self, role, name, track, ctx, now):
        opts = _options(role, ctx, track)
        t = phrase(ctx['target'])
        questions = {
            'is_target': {'type': 'noul', 'instructions': (
                f'Is `candidate` really a {t}? Weigh `candidate.vlm_checks` most, since it is '
                f'independent of the detector; then views from different positions '
                f'(`candidate.views`, `candidate.bearing_span_rad`, `candidate.closest_view_m`); then '
                f'whether `candidate.objects_seen_near_it` fit a room where a {t} is found. Remember '
                f'`detector.reliability`.')},
            'action': {'type': 'choice', 'instructions': (
                f'What should `this_robot` do about `candidate` now, to get a real {t} confirmed by the '
                f'dog soonest without wasting the dog\'s time or a drone\'s battery?'),
                'criteria': opts},
        }
        t0 = time.perf_counter()
        a = self.client.judge(_state(role, name, track, ctx, now), questions)
        latency = time.perf_counter() - t0
        p = float(a['is_target']['value'])
        chosen = a['action']['choice']
        return _gate(chosen, p, opts, track, ctx['target']), latency, {
            'policy': 'jev', 'p_target': round(p, 4), 'model_choice': chosen, 'offered': sorted(opts),
            'probabilities': a['action']['probabilities'], 'usage': getattr(self.client, 'last_usage', None)}

    def frontier(self, name, fronts, tasks, world, target, now):
        opts, crit = self._frontier_state(name, fronts, tasks, world, target)
        if not opts:
            return None, 0.0, {'policy': 'jev', 'note': 'no reachable frontier'}
        if len(opts) == 1:
            return opts[0]['id'], 0.0, {'policy': 'jev', 'note': 'single option'}
        t0 = time.perf_counter()
        a = self.client.judge({'mission': f'Searching an unknown building for a {phrase(target)}.',
                               'this_robot': name},
                              {'frontier': {'type': 'choice', 'criteria': crit, 'instructions': (
                                  f'Which opening should `this_robot` explore next to find a '
                                  f'{phrase(target)} soonest? Prefer openings whose nearby objects suggest '
                                  f'the room a {phrase(target)} is usually in; avoid ones a teammate is '
                                  f'heading to; otherwise prefer the shorter trip.')}})
        latency = time.perf_counter() - t0
        return a['frontier']['choice'], latency, {'policy': 'jev',
                                                  'probabilities': a['frontier']['probabilities']}


VLM_LOCAL_PROMPT = """You decide one step for one robot of a search team. Here is the situation as JSON:

{state}

Possible actions:
{options}

Answer with JSON only, in exactly this form:
{{"is_target": <probability between 0 and 1 that the candidate really is a {target}>, "action": "<one action name from the list>"}}"""

VLM_FRONTIER_PROMPT = """A search robot is looking for a {target} in an unknown building and needs a new opening to explore. Options:
{options}

Prefer openings whose nearby objects suggest the room a {target} is usually in; avoid ones a teammate is heading to; otherwise prefer the shorter trip. Answer with JSON only: {{"frontier": "<id>"}}"""


class VLMDecider(_ModelDecider):
    """The same local decisions asked of the VLM instead of Jev (text only).

    Its latency is charged like any other call and it queues on the same
    single-sequence server as the brain and the second opinions."""
    kind = 'vlm'

    def __init__(self, server):
        self.server = server

    def _ask(self, text):
        payload = {'model': 'qwen3vl', 'messages': [{'role': 'user', 'content': text}],
                   'max_tokens': 60, 'temperature': 0.0, 'seed': 0}
        t0 = time.perf_counter()
        r = self.server.session.post(self.server.endpoint, json=payload, timeout=120)
        r.raise_for_status()
        latency = time.perf_counter() - t0
        self.server.calls += 1
        raw = r.json()['choices'][0]['message']['content']
        m = re.search(r'\{.*\}', raw, re.S)
        doc = json.loads(m.group(0)) if m else {}
        return doc, raw, latency

    def candidate(self, role, name, track, ctx, now):
        opts = _options(role, ctx, track)
        text = VLM_LOCAL_PROMPT.format(state=json.dumps(_state(role, name, track, ctx, now)),
                                       options='\n'.join(f'- {k}: {v}' for k, v in opts.items()),
                                       target=phrase(ctx['target']))
        doc, raw, latency = self._ask(text)
        try:
            p = float(doc.get('is_target', 0.0))
        except (TypeError, ValueError):
            p = 0.0
        chosen = str(doc.get('action', 'continue_task'))
        if chosen not in opts:
            chosen = 'continue_task'
        return _gate(chosen, p, opts, track, ctx['target']), latency, {'policy': 'vlm', 'p_target': round(p, 4),
                                                 'model_choice': chosen, 'offered': sorted(opts),
                                                 'raw': raw[:300]}

    def frontier(self, name, fronts, tasks, world, target, now):
        opts, crit = self._frontier_state(name, fronts, tasks, world, target)
        if not opts:
            return None, 0.0, {'policy': 'vlm', 'note': 'no reachable frontier'}
        if len(opts) == 1:
            return opts[0]['id'], 0.0, {'policy': 'vlm', 'note': 'single option'}
        doc, raw, latency = self._ask(VLM_FRONTIER_PROMPT.format(
            target=phrase(target), options='\n'.join(f'- {k}: {v}' for k, v in crit.items())))
        fid = str(doc.get('frontier', '')).strip().upper()
        if fid not in crit:
            fid = opts[0]['id']
        return fid, latency, {'policy': 'vlm', 'raw': raw[:200]}


def make_decider(kind, jev=None, server=None):
    if kind == 'rule':
        return RuleDecider()
    if kind == 'jev':
        return JevDecider(jev)
    if kind == 'vlm':
        return VLMDecider(server)
    raise ValueError(kind)

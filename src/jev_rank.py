"""Optional allocation `jev`: the judge, instead of the VLM, ranks the frontiers.

One request per ranking: the state lists the target and, for each frontier, the
objects detected near it; one Noul question per frontier asks whether the target
is likely beyond it. Code sorts by probability and hands the order to the same
Hungarian assignment the VLM ranking uses. Jev judges evidence (where is the
target likely?), never picks a robot or a frontier itself.
"""
from __future__ import annotations


def rank_frontiers(client, target, frontiers):
    """frontiers: [{'id': 'F3', 'landmarks_near': [...], 'width_m': 1.2}, ...]
    -> (ids ordered most promising first, {id: probability})."""
    if not frontiers:
        return [], {}
    t = target.replace('_', ' ')
    state = {'task': f'A robot team is searching an unknown building for a {t}. Each frontier is an opening '
                     f'into space the team has not searched yet.',
             'frontiers': {f['id']: {'objects_seen_near_it': f.get('landmarks_near', []) or ['nothing detected'],
                                     'opening_width_m': f.get('width_m')} for f in frontiers}}
    questions = {f['id']: {'type': 'noul', 'instructions': (
        f'Judging from the objects seen near frontier {f["id"]}, is a {t} likely to be found beyond it?'),
        'criteria': {'true': f'the nearby objects suggest the kind of room where a {t} usually is, or lead '
                             f'towards it',
                     'false': f'the nearby objects suggest a room where a {t} is unusual'}}
        for f in frontiers}
    ans = client.judge(state, questions)
    p = {fid: float(a['value']) for fid, a in ans.items()}
    return sorted(p, key=lambda k: -p[k]), p


if __name__ == '__main__':
    class Fake:                     # ranks by whether 'sink' is near, to check the plumbing
        def judge(self, state, questions):
            return {k: {'value': 0.9 if 'sink' in state['frontiers'][k]['objects_seen_near_it'] else 0.2}
                    for k in questions}
    order, p = rank_frontiers(Fake(), 'toilet', [{'id': 'F1', 'landmarks_near': ['sofa']},
                                                  {'id': 'F2', 'landmarks_near': ['sink', 'towel']}])
    assert order == ['F2', 'F1'] and p['F2'] > p['F1'], (order, p)
    print('ok', order, p)

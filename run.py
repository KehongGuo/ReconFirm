#!/usr/bin/env python3
"""Run an experiment from a config: every (pass, episode, condition) mission, one process each, resumable.

  python run.py -c configs/mp3d_table.yaml                     # the large-building table (Table I)
  python run.py -c configs/hm3d_val.yaml gpu=1 vlm.url=http://localhost:8011
  python run.py -c configs/mp3d_table.yaml --dry-run           # print the commands only

A mission whose directory holds summary.json is done and skipped, so a stopped run resumes where it was. Several
runners may share one experiment (e.g. one per GPU, each with its own VLM server): each mission is claimed with a lock
file. A mission that crashed is retried up to retries.crash times; one whose model calls failed (service errors) is
moved aside and retried, and three such missions in a row stop the runner (NEEDS_ATTENTION in the output directory).
Stop after the current mission: touch out/<experiment>/STOP.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def load_config(path, overrides=()):
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    if cfg.get('base'):
        base = load_config(Path(path).parent / cfg.pop('base'))
        cfg = merge(base, cfg)
    for item in overrides:                       # key.sub=value, value parsed as YAML
        key, _, value = item.partition('=')
        node = cfg
        *parents, leaf = key.split('.')
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return out


def environment(cfg):
    env = dict(os.environ)
    here = lambda p: str(ROOT / p)               # relative paths are relative to the repository
    env.update({'SCOUT_MP3D_ROOT': here(cfg['data']['mp3d_root']), 'SCOUT_HM3D_ROOT': here(cfg['data']['hm3d_root']),
                'SCOUT_OBJECTNAV_ROOT': here(cfg['data']['objectnav_root']),
                'SCOUT_VLM_URL': cfg['vlm']['url'], 'SCOUT_VLM_MODEL': cfg['vlm']['model'],
                'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'})
    for key, var in (('key_file', 'TYPESAFE_API_KEY_FILE'), ('log', 'SCOUT_JEV_LOG'), ('pin_addrs', 'SCOUT_JEV_ADDRS')):
        if cfg['judge'].get(key):
            env[var] = here(cfg['judge'][key]) if key != 'pin_addrs' else str(cfg['judge'][key])
    if cfg['verifier'].get('model'):
        env['SCOUT_VERIFIER_MODEL'] = cfg['verifier']['model']
    if cfg.get('save_crops'):
        env['SCOUT_CROPS'] = '1'
    if cfg.get('gpu') is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(cfg['gpu'])
    return env


def command(cfg, cond, episode_id, out):
    c = CONDITIONS[cond]
    entry = ROOT / 'src' / ('hm3d_eval.py' if cfg['dataset'] == 'hm3d' else 'team_run.py')
    cmd = [sys.executable, str(entry), '--episodes', str(ROOT / cfg['episodes']), '--episode-id', episode_id,
           '--allocation', c['allocation'], '--drone-local', c['drone_local'], '--dog-local', c['dog_local'],
           '--clock', c['clock'], '--scheduler', cfg['scheduler'], '--mission-s', str(cfg['mission_s']),
           '--localize', cfg['localize'], '--out', str(out)]
    if 'jev' in (c['allocation'], c['drone_local'], c['dog_local']):
        cmd.append('--allow-paid-inference')
    if cfg['memory'].get('store'):
        cmd += ['--experience', str(ROOT / cfg['memory']['store'])] + (['--experience-learn'] if cfg['memory'].get('learn') else [])
    if cfg.get('video'):
        cmd.append('--video')
    return cmd


def run_mission(cfg, cmd, run, env, crashes, aside, note):
    """One mission in its own process -> 'done', 'crash' or 'service_error' (moved aside for a rerun)."""
    while not vlm_up(cfg['vlm']['url']):
        note(f"waiting for the VLM server at {cfg['vlm']['url']}")
        time.sleep(30)
    shutil.rmtree(run, ignore_errors=True)
    with open(f'{run}.log', 'w') as fh:
        rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT / 'src').returncode
    if not (run / 'summary.json').exists():
        Path(f'{run}.log').rename(f'{run}.crash{crashes}.log')
        note(f'{run.name} crashed rc={rc}')
        return 'crash'
    s = json.loads((run / 'summary.json').read_text())
    if s['counts'].get('service_errors'):
        run.rename(f'{run}.svcerr{aside}')
        Path(f'{run}.log').rename(f'{run}.svcerr{aside}.log')
        note(f"{run.name} service errors={s['counts']['service_errors']}, moved aside")
        return 'service_error'
    note(f"{run.name} {s['outcome']}")
    return 'done'


def vlm_up(url):
    import urllib.request
    try:
        urllib.request.urlopen(url + '/v1/models', timeout=5).read()
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('overrides', nargs='*', help='key.sub=value')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    cfg = merge(load_config(ROOT / 'configs' / 'default.yaml'), load_config(a.config, a.overrides))
    global CONDITIONS
    CONDITIONS = yaml.safe_load((ROOT / 'configs' / 'conditions.yaml').read_text())
    unknown = [c for c in cfg['conditions'] if c not in CONDITIONS]
    if unknown:
        raise SystemExit(f'unknown conditions {unknown}; see configs/conditions.yaml')
    episodes = json.loads((ROOT / cfg['episodes']).read_text())['episodes']
    ids = [e['episode_id'] for e in episodes] if cfg['episode_ids'] == 'all' else list(cfg['episode_ids'])
    top = ROOT / 'out' / cfg['experiment']
    top.mkdir(parents=True, exist_ok=True)
    (top / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    env, log = environment(cfg), top / 'run.log'

    def note(msg):
        with open(log, 'a') as fh:
            fh.write(f"{time.strftime('%F %T')} {msg}\n")

    service_run = 0
    for p in range(1, cfg['passes'] + 1):
        out = top / f'p{p}' if cfg['passes'] > 1 else top
        out.mkdir(exist_ok=True)
        for attempt_round in range(2):           # the second round retries crashed / moved-aside missions
            for episode_id in ids:
                for cond in cfg['conditions']:
                    if (top / 'STOP').exists():
                        note('stopped by STOP file')
                        return
                    run = out / f"{episode_id.replace(':', '_')}__{cond}"
                    if (run / 'summary.json').exists():
                        continue
                    crashes = sum(1 for _ in out.glob(f'{run.name}.crash*'))
                    aside = sum(1 for _ in out.glob(f'{run.name}.svcerr*'))
                    if crashes >= cfg['retries']['crash'] or aside >= cfg['retries']['service_error']:
                        continue
                    cmd = command(cfg, cond, episode_id, run)
                    if a.dry_run:
                        print(' '.join(cmd))
                        continue
                    with open(f'{run}.lock', 'w') as lock:
                        try:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            continue                     # another runner has it
                        if (run / 'summary.json').exists():
                            continue                     # it finished while we looked
                        outcome = run_mission(cfg, cmd, run, env, crashes, aside, note)
                    if outcome == 'service_error':
                        service_run += 1
                        if service_run >= 3:
                            (top / 'NEEDS_ATTENTION').write_text('model calls keep failing (judge credit or network?)\n')
                            note('NEEDS_ATTENTION: model calls keep failing')
                            return
                    elif outcome == 'done':
                        service_run = 0
    note('all done')


if __name__ == '__main__':
    main()

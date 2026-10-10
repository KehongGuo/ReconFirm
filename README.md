# ReconFirm

**Scout Fast, Commit Carefully: Commitment-Aware Scheduling for Map-Free Aerial–Legged Object Search**

Drones *recon*, the legged robot *confirms*. Commitment-aware scheduling for map-free aerial–legged object search. Three drones scout an unknown building while a
legged robot, the only robot that may declare the target found, waits to be committed. A VLM ranks frontiers and
proposes tasks from evidence-backed menus, a Hungarian assignment routes the robots, and a calibrated evidence judge
admits each irreversible commitment (dispatching the legged robot, declaring the target, discarding a candidate) only
when its risk is justified. Every model call is charged its measured latency on the mission clock.

This repository holds the full system, the large-building (Matterport3D) experiments, the HM3D ObjectNav evaluation,
and the self-evolution pipeline (case memory and evolving verifier).

## Layout

```
run.py                 the entry point: runs an experiment from a config (resumable, multi-GPU)
configs/               default.yaml (every setting), conditions.yaml (the team conditions), experiments
src/                   the system
  team_run.py            one mission: scheduler, commitment gates, clock, logging, scoring
  team_world.py          the team's world: robots, batteries and docking, detection, candidate tracks
  team_agents.py         VLM scheduler prompts, deciders (judge + two-key gates, rules), task menus
  aerial_map.py          two-layer observed map (ground layer for the legged robot, air layer for the drones)
  team_board.py          the board the VLM reads (rendered map + structured summary)
  search_agents.py       VLM server client, verifier call
  verifier_prompt.py     the verifier's question and answer parser
  perception.py          Grounding DINO detector;  mask_localize.py  SAM-mask placement of detections
  jev.py                 the evidence judge client (TypeSafe Jev)
  team_experience_v3.py  the case memory (build / freeze / check, and its hook at the declaration)
  hm3d_eval.py           HM3D ObjectNav wrapper (benchmark agent, naming standard, scoring)
tools/                 table.py (Table I), hm3d_summary.py (HM3D SR/SPL), verifier_evolve.py (self-evolving verifier),
                       make_hm3d_episodes.py
data/                  episode files, the HM3D naming standard, the verifier teacher's instruction
scripts/               serve_vllm.sh, get_pope.sh
tests/                 unit tests (python -m unittest discover -s tests -t .)
```

## Installation

Three environments, because the simulator, the VLM server and verifier training pin different library versions.

```bash
# 1. simulation and agents (Python 3.9)
conda create -n scout python=3.9 && conda activate scout
conda install -c aihabitat -c conda-forge habitat-sim=0.3.3 withbullet headless
pip install -r requirements.txt

# 2. the VLM server (Python 3.11)
conda create -n scout-vllm python=3.11 && conda activate scout-vllm
pip install -r requirements-vllm.txt

# 3. optional, verifier self-evolution (Python 3.11)
conda create -n scout-train python=3.11 && conda activate scout-train
pip install -r requirements-train.txt
```

Every experiment in the paper ran on 16 GB consumer GPUs (RTX 4070 Ti SUPER, RTX 5080): the VLM server takes about
11.5 GB and the simulator with the detector and SAM about 3-4 GB, so large buildings run with the server on one GPU and
the simulator on another.

## Data and models

* **Matterport3D** for Habitat (the large-building test set: 10 held-out buildings, `data/episodes/team_episodes_v2_test.json`).
  Request access at https://niessner.github.io/Matterport/ and download with Habitat's `datasets_download`.
* **HM3D v0.2 with semantics** and the **ObjectNav HM3D v2** episodes (https://aihabitat.org/datasets/hm3d-semantics/).
* Set the three roots in a config's `data:` section (defaults: `data/scene_datasets`, `data/scene_datasets/hm3d`,
  `data/datasets/objectnav/hm3d/objectnav_hm3d_v2`).
* **VLM:** Qwen3-VL-8B-Instruct-FP8 served by vLLM (`scripts/serve_vllm.sh`). Grounding DINO (base) and SAM (ViT-B)
  download from the Hugging Face hub on first use.
* **Evidence judge:** TypeSafe Jev 1.13 (https://typesafe.ai), a paid text-only API. Export `TYPESAFE_API_KEY` (or set
  `judge.key_file`). Conditions that use rules only need no key. In our runs the judge cost about USD 0.06 per 1000
  requests.

## Running

```bash
conda activate scout-vllm && scripts/serve_vllm.sh 0 8010 &      # GPU 0, port 8010
conda activate scout
python run.py -c configs/hm3d_val.yaml gpu=1                     # HM3D val, 1000 episodes
python run.py -c configs/mp3d_table.yaml gpu=1                   # Table I: 8 conditions x 50 missions x 2 passes
python tools/hm3d_summary.py out/hm3d_val
python tools/table.py out/mp3d_table
```

Any config key can be overridden on the command line (`mission_s=300`, `vlm.url=http://localhost:8011`,
`"episode_ids=[i5noydFURQK:v2s20]"`, `"conditions=[ours,frontier_rules]"`). `--dry-run` prints the mission commands.
Each mission runs in its own process and writes `out/<experiment>/<episode>__<condition>/` (`events.jsonl`, the board
images, `summary.json`, `manifest.json` with the source hashes). A stopped run resumes where it left off, and several
runners (for example one per GPU, each with its own VLM server and `vlm.url`) can share one experiment.

### Conditions

| name | allocation | drone decisions | legged-robot decisions | clock |
|---|---|---|---|---|
| `ours` | VLM ranks + assignment | judge + gates | judge + gates | charged |
| `frontier_rules` | geometric | rules | rules | charged |
| `vlm_direct` | VLM assigns every robot | judge + gates | judge + gates | charged |
| `wo_gates` | VLM ranks + assignment | rules | rules | charged |
| `wo_semantic_scheduler` | geometric | judge + gates | judge + gates | charged |
| `gates_legged_only` | VLM ranks + assignment | rules | judge + gates | charged |
| `gates_drones_only` | VLM ranks + assignment | judge + gates | rules | charged |
| `latency_free` | VLM ranks + assignment | judge + gates | judge + gates | free |

## Self-evolution

The judge never learns; the case memory and the verifier do, on HM3D train scenes split into folds R1/R2/R3 (one per
round), B (memory veto gate) and C (practice fold that scores every round); validation episodes are never used.

```bash
python run.py -c configs/practice_round.yaml                               # search R1: learn memory, save crops
python src/team_experience_v3.py freeze memory/store.json --out memory/frozen.json [--veto-gate 0.3]
python tools/verifier_evolve.py --help                                     # one verifier round, step by step
python run.py -c configs/hm3d_val300_evolved.yaml                          # frozen memory + evolved verifier
```

The verifier's corrections are written by a teacher model following `data/teacher_instruction.md`; labels always come
from the scene annotation, never from the teacher.

## Tests

```bash
python -m unittest discover -s tests -t .
```

## Citation

```bibtex
@article{guo2026scout,
  title  = {Scout Fast, Commit Carefully: Commitment-Aware Scheduling for Map-Free Aerial--Legged Object Search},
  author = {Guo, Kehong},
  year   = {2026}
}
```

## License

MIT (see `LICENSE`). Matterport3D, HM3D, Qwen3-VL, Grounding DINO, SAM and the Jev API are used under their own terms.

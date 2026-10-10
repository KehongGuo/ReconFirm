"""Where the scene datasets live, and how to find one scene.

SCOUT_MP3D_ROOT: the directory holding Matterport3D for Habitat (a `mp3d*/` subdirectory with the scene dataset config
and one directory per building). SCOUT_HM3D_ROOT: HM3D v0.2 with semantics (`hm3d_annotated_basis.scene_dataset_config
.json` and the `val/`, `train/` scene directories). run.py sets both from the config's `data` section.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

MP3D_ROOT = os.environ.get("SCOUT_MP3D_ROOT", "data/scene_datasets")
HM3D_ROOT = os.environ.get("SCOUT_HM3D_ROOT", "data/scene_datasets/hm3d")


@dataclass
class SceneRef:
    scene_id: str
    dataset_config: str | None
    dataset: str

    def __str__(self) -> str:
        return f"{self.dataset}:{os.path.basename(self.scene_id)}"


def _mp3d_config(root: str) -> str | None:
    for cand in (f"{root}/mp3d.scene_dataset_config.json", f"{root}/mp3d_example.scene_dataset_config.json"):
        if os.path.exists(cand):
            return cand
    found = glob.glob(f"{root}/*.scene_dataset_config.json")
    return found[0] if found else None


def mp3d_scenes() -> list[SceneRef]:
    """Every MP3D building under MP3D_ROOT, deduplicated by building id (a full release wins over the example set)."""
    dirs = []
    for root in glob.glob(f"{MP3D_ROOT}/mp3d*"):
        if _mp3d_config(root):
            dirs += [d.rstrip("/") for d in sorted(glob.glob(f"{root}/*/"))
                     if os.path.exists(os.path.join(d, f"{os.path.basename(d.rstrip('/'))}.glb"))]
    by_name: dict[str, SceneRef] = {}
    for d in sorted(dirs, key=lambda d: -len(glob.glob(f"{os.path.dirname(d)}/*/"))):
        name = os.path.basename(d)
        if name not in by_name:
            by_name[name] = SceneRef(f"{d}/{name}.glb", _mp3d_config(os.path.dirname(d)), "mp3d")
    return [by_name[k] for k in sorted(by_name)]


def hm3d_scene(scene_id: str) -> SceneRef | None:
    """HM3D v0.2 (semantic) scene by its short id, e.g. 'zt1RVoi7PcG'."""
    hits = glob.glob(f"{HM3D_ROOT}/*/*-{scene_id}/{scene_id}.basis.glb")
    if not hits:
        return None
    return SceneRef(hits[0], f"{HM3D_ROOT}/hm3d_annotated_basis.scene_dataset_config.json", "hm3d")

"""Resume docking against the existing manifest, skipping completed ligands."""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import pandas as pd

from vina_rescore.config import Config
from vina_rescore.data import docking_box
from vina_rescore.prep import prepare_receptor
from vina_rescore.docking import precompute_maps, dock_batch, DockParams


def main() -> None:
    cfg = Config().resolve()
    manifest = pd.read_csv(cfg.work_dir / "manifest.csv")
    print(f"[data] manifest: {len(manifest)} ligands "
          f"({int(manifest.label.sum())} actives / {int((manifest.label==0).sum())} decoys)")

    receptor_pdbqt, receptor_h = prepare_receptor(cfg)
    center, size = docking_box(cfg)
    maps_prefix = precompute_maps(cfg, receptor_pdbqt, center, size)
    print(f"[maps] {maps_prefix}")

    params = DockParams(
        maps_prefix=maps_prefix, center=center, size=size,
        exhaustiveness=cfg.exhaustiveness, n_poses=cfg.n_poses,
        cpu_per_worker=cfg.cpu_per_worker, embed_seed=cfg.embed_seed,
    )
    dock_batch(cfg, manifest, params, resume=True)
    print("[done] resume complete")


if __name__ == "__main__":
    main()

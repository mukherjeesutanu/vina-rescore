"""Driver: prepare receptor, build manifest, precompute maps, dock the batch."""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

from vina_rescore.config import Config
from vina_rescore.data import build_manifest, docking_box
from vina_rescore.prep import prepare_receptor
from vina_rescore.docking import precompute_maps, dock_batch, DockParams


def main() -> None:
    cfg = Config().resolve()
    cfg.to_json(cfg.out_dir / "run_config.json")

    manifest = build_manifest(cfg)
    print(f"[data] manifest: {len(manifest)} ligands "
          f"({int(manifest.label.sum())} actives / {int((manifest.label==0).sum())} decoys)")

    receptor_pdbqt, receptor_h = prepare_receptor(cfg)
    print(f"[prep] receptor -> {receptor_pdbqt.name}, {receptor_h.name}")

    center, size = docking_box(cfg)
    print(f"[box ] center={tuple(round(c,2) for c in center)} size={size}")

    maps_prefix = precompute_maps(cfg, receptor_pdbqt, center, size)
    print(f"[maps] {maps_prefix}")

    params = DockParams(
        maps_prefix=maps_prefix, center=center, size=size,
        exhaustiveness=cfg.exhaustiveness, n_poses=cfg.n_poses,
        cpu_per_worker=cfg.cpu_per_worker, embed_seed=cfg.embed_seed,
    )
    scores = dock_batch(cfg, manifest, params)
    print("[done] wrote", cfg.out_dir / "vina_scores.csv",
          "and", cfg.work_dir / "docked_poses.sdf")


if __name__ == "__main__":
    main()

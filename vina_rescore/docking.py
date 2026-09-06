"""Parallel AutoDock Vina docking engine.

Affinity maps are computed once for the receptor/box, then reused by every
worker (``load_maps``) so per-ligand cost is dominated by the search itself.
Each worker embeds and prepares its ligand, docks it, and returns the top
pose (as an RDKit MolBlock) plus its Vina affinity. The parent assembles a
combined ``docked_poses.sdf`` and a ``vina_scores.csv`` table.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from rdkit import Chem

from .config import Config
from .prep import prepare_ligand_pdbqt, PreparationError


@dataclass(frozen=True)
class DockParams:
    """Lightweight, picklable bundle of per-worker docking parameters."""

    maps_prefix: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    exhaustiveness: int
    n_poses: int
    cpu_per_worker: int
    embed_seed: int


def precompute_maps(cfg: Config, receptor_pdbqt: Path,
                    center: tuple[float, float, float],
                    size: tuple[float, float, float]) -> str:
    """Compute and cache Vina affinity maps for the receptor/box once.

    Returns the map-file prefix that workers pass to ``Vina.load_maps``.
    """
    from vina import Vina

    import glob

    maps_dir = cfg.work_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(maps_dir / "receptor")
    if glob.glob(prefix + ".*.map"):
        # Maps already computed for this receptor/box — reuse them.
        return prefix
    v = Vina(sf_name="vina", cpu=0, seed=cfg.random_seed, verbosity=0)
    v.set_receptor(str(receptor_pdbqt))
    v.compute_vina_maps(center=list(center), box_size=list(size))
    v.write_maps(prefix)
    return prefix


def _dock_one(task: tuple[str, str], params: DockParams) -> dict[str, object]:
    """Worker: prepare + dock a single ligand. Never raises; errors captured."""
    ligand_id, smiles = task
    out: dict[str, object] = {
        "ligand_id": ligand_id, "vina_affinity": None,
        "molblock": None, "error": None,
    }
    try:
        from vina import Vina
        from meeko import PDBQTMolecule, RDKitMolCreate

        pdbqt = prepare_ligand_pdbqt(smiles, seed=params.embed_seed)

        v = Vina(sf_name="vina", cpu=params.cpu_per_worker, seed=42, verbosity=0)
        v.load_maps(params.maps_prefix)
        v.set_ligand_from_string(pdbqt)
        v.dock(exhaustiveness=params.exhaustiveness, n_poses=params.n_poses)

        out["vina_affinity"] = float(v.energies(n_poses=1)[0][0])

        pose_pdbqt = v.poses(n_poses=1)
        pmol = PDBQTMolecule(pose_pdbqt, is_dlg=False, skip_typing=True)
        rdmols = RDKitMolCreate.from_pdbqt_mol(pmol)
        lig = rdmols[0]
        lig.SetProp("_Name", ligand_id)
        lig.SetProp("vina_affinity", f"{out['vina_affinity']:.3f}")
        out["molblock"] = Chem.MolToMolBlock(lig)
    except PreparationError as exc:
        out["error"] = f"prep: {exc}"
    except Exception as exc:  # noqa: BLE001 - defensive: keep batch alive
        out["error"] = f"dock: {type(exc).__name__}: {exc}"
    return out


def _already_done(sdf_path: Path, scores_path: Path) -> set[str]:
    """Ligand IDs already recorded (in the SDF or the scores CSV)."""
    done: set[str] = set()
    if sdf_path.exists():
        supp = Chem.SDMolSupplier(str(sdf_path), sanitize=False)
        for m in supp:
            if m is not None and m.HasProp("_Name"):
                done.add(m.GetProp("_Name"))
    if scores_path.exists():
        try:
            done |= set(pd.read_csv(scores_path)["ligand_id"].astype(str))
        except Exception:
            pass
    return done


def _append_pose(sdf_path: Path, ligand_id: str, affinity: float, molblock: str) -> None:
    """Durably append one pose (with properties) to the master SDF."""
    from io import StringIO
    mol = Chem.MolFromMolBlock(molblock, removeHs=False)
    if mol is None:
        return
    mol.SetProp("_Name", ligand_id)
    mol.SetProp("vina_affinity", f"{affinity:.3f}")
    buf = StringIO()
    w = Chem.SDWriter(buf)
    w.write(mol)
    w.close()
    with open(sdf_path, "a") as fh:
        fh.write(buf.getvalue())
        fh.flush()
        os.fsync(fh.fileno())


def _append_score(scores_path: Path, row: dict[str, object]) -> None:
    """Durably append one row to the scores CSV (writing a header if new)."""
    new = not scores_path.exists()
    with open(scores_path, "a") as fh:
        if new:
            fh.write("ligand_id,label,vina_affinity,error\n")
        aff = "" if row["vina_affinity"] is None else f"{row['vina_affinity']:.3f}"
        err = "" if row["error"] is None else str(row["error"]).replace(",", ";").replace("\n", " ")
        fh.write(f"{row['ligand_id']},{row['label']},{aff},{err}\n")
        fh.flush()
        os.fsync(fh.fileno())


def dock_batch(cfg: Config, manifest: pd.DataFrame, params: DockParams,
               progress_every: int = 10, resume: bool = True) -> pd.DataFrame:
    """Dock ligands in parallel, resumably, with durable per-ligand writes.

    Each completed pose is appended to ``<work_dir>/docked_poses.sdf`` and each
    result row to ``<out_dir>/vina_scores.csv`` immediately (flushed + fsync'd),
    so an interrupted run loses no completed work. With ``resume=True`` ligands
    already present in either file are skipped.
    """
    sdf_path = cfg.work_dir / "docked_poses.sdf"
    scores_path = cfg.out_dir / "vina_scores.csv"
    label_map = dict(zip(manifest["ligand_id"], manifest["label"]))

    done = _already_done(sdf_path, scores_path) if resume else set()
    tasks = [(lid, smi) for lid, smi in zip(manifest["ligand_id"], manifest["smiles"])
             if lid not in done]
    print(f"[dock] {len(done)} already done; docking {len(tasks)} remaining", flush=True)

    t0 = time.time()
    n = 0
    with ProcessPoolExecutor(max_workers=cfg.n_workers) as ex:
        futs = {ex.submit(_dock_one, t, params): t[0] for t in tasks}
        for fut in as_completed(futs):
            res = fut.result()
            n += 1
            mb = res.get("molblock")
            if mb is not None and res["vina_affinity"] is not None:
                _append_pose(sdf_path, str(res["ligand_id"]),
                             float(res["vina_affinity"]), mb)
            _append_score(scores_path, {
                "ligand_id": res["ligand_id"],
                "label": label_map.get(res["ligand_id"], ""),
                "vina_affinity": res["vina_affinity"], "error": res["error"],
            })
            if n % progress_every == 0 or n == len(tasks):
                rate = n / (time.time() - t0)
                print(f"[dock] {n}/{len(tasks)} "
                      f"({rate:.3f}/s, {(len(tasks)-n)/max(rate,1e-9):.0f}s left)",
                      flush=True)

    scores = pd.read_csv(scores_path)
    n_ok = int(scores["vina_affinity"].notna().sum())
    print(f"[dock] complete: {n_ok}/{len(scores)} docked, elapsed {time.time()-t0:.0f}s")
    return scores

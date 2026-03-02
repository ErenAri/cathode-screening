"""Dataset and collation for MACE-based training.

Loads pre-cached structure files (atomic_numbers, positions, cell) and converts
them on-the-fly to MACE's expected input format: one-hot node_attrs, ASE-based
neighbor lists, and periodic shift vectors.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _compute_neighbors(
    positions: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute periodic neighbor list via ASE.

    Returns:
        edge_src, edge_dst  – integer arrays [E]
        cart_shifts          – Cartesian shift vectors [E, 3]
        unit_shifts          – integer cell-coordinate shifts [E, 3]
    """
    try:
        from ase import Atoms
        from ase.neighborlist import neighbor_list as ase_nl
    except ImportError:
        raise ImportError("ASE is required for MACE data prep: pip install ase")

    atoms = Atoms(
        positions=positions,
        cell=cell,
        pbc=True,
        numbers=np.ones(len(positions), dtype=int),  # placeholder
    )
    edge_src, edge_dst, unit_shifts = ase_nl("ijS", atoms, cutoff=cutoff)
    cart_shifts = unit_shifts @ cell  # [E, 3]
    return (
        edge_src.astype(np.int64),
        edge_dst.astype(np.int64),
        cart_shifts.astype(np.float64),
        unit_shifts.astype(np.float64),
    )


class StructureDataset(Dataset):
    """Load cached crystal structures and prepare MACE-compatible tensors.

    Each sample is a ``(data_dict, y)`` pair where ``data_dict`` contains
    positions, one-hot node attributes, edge indices, periodic shifts, and cell.

    Parameters
    ----------
    df : DataFrame
        Must contain ``material_id`` and the *target_col*.
    target_col : str
        Column name for the regression target (e.g. ``energy_above_hull``).
    structure_dir : path
        Directory of ``.npz`` files keyed by material_id, each containing
        ``atomic_numbers``, ``positions``, ``cell``.
    r_max : float
        Cutoff radius for MACE neighbor list (default 5.0 A).
    z_table : list[int] | None
        Ordered atomic numbers for one-hot encoding.  If *None*, uses Z 1-94.
    """

    def __init__(
        self,
        df,
        target_col: str,
        structure_dir: str | Path,
        r_max: float = 5.0,
        z_table: Optional[List[int]] = None,
        default_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.target_col = target_col
        self.structure_dir = Path(structure_dir)
        self.r_max = r_max
        self.z_table = z_table or list(range(1, 95))
        self._z_to_idx = {z: i for i, z in enumerate(self.z_table)}
        self.dtype = default_dtype

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor | int], torch.Tensor]:
        row = self.df.iloc[idx]
        mid = row["material_id"]
        y = torch.tensor(float(row[self.target_col]), dtype=torch.float32)

        cache_path = self.structure_dir / f"{mid}.npz"
        with np.load(cache_path) as z:
            atomic_numbers = z["atomic_numbers"]
            positions = z["positions"].astype(np.float64)
            cell = z["cell"].astype(np.float64)

        # Neighbor list
        edge_src, edge_dst, cart_shifts, unit_shifts = _compute_neighbors(
            positions, cell, self.r_max
        )

        # One-hot encode atomic numbers using the model's z_table
        n_atoms = len(atomic_numbers)
        n_types = len(self.z_table)
        node_attrs = np.zeros((n_atoms, n_types), dtype=np.float64)
        for i, z_val in enumerate(atomic_numbers):
            z_idx = self._z_to_idx.get(int(z_val))
            if z_idx is not None:
                node_attrs[i, z_idx] = 1.0
            else:
                logger.warning("Unknown element Z=%d in %s, skipping one-hot", z_val, mid)

        return {
            "positions": torch.tensor(positions, dtype=self.dtype),
            "node_attrs": torch.tensor(node_attrs, dtype=self.dtype),
            "edge_index": torch.tensor(
                np.stack([edge_src, edge_dst], axis=0), dtype=torch.long
            ),
            "shifts": torch.tensor(cart_shifts, dtype=self.dtype),
            "unit_shifts": torch.tensor(unit_shifts, dtype=self.dtype),
            "cell": torch.tensor(cell, dtype=self.dtype),  # [3, 3]
            "n_atoms": n_atoms,
        }, y


def mace_collate(
    batch: List[Tuple[Dict[str, torch.Tensor | int], torch.Tensor]],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Batch multiple MACE structure dicts (+ targets) into a single dict.

    Follows the same offset-based batching as PyG / MACE's internal Batch:
    edge indices are offset by cumulative node counts, cells are stacked,
    and a ``batch`` tensor maps each node to its graph index.
    """
    data_list, targets = zip(*batch)

    positions_list: list = []
    node_attrs_list: list = []
    edge_index_list: list = []
    shifts_list: list = []
    unit_shifts_list: list = []
    cells: list = []
    batch_idx: list = []
    ptr = [0]

    node_offset = 0
    for i, d in enumerate(data_list):
        n = d["n_atoms"]
        positions_list.append(d["positions"])
        node_attrs_list.append(d["node_attrs"])

        ei = d["edge_index"].clone()
        ei += node_offset
        edge_index_list.append(ei)

        shifts_list.append(d["shifts"])
        unit_shifts_list.append(d["unit_shifts"])
        cells.append(d["cell"])

        batch_idx.append(torch.full((n,), i, dtype=torch.long))
        node_offset += n
        ptr.append(node_offset)

    batched: Dict[str, torch.Tensor] = {
        "positions": torch.cat(positions_list, dim=0),
        "node_attrs": torch.cat(node_attrs_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "shifts": torch.cat(shifts_list, dim=0),
        "unit_shifts": torch.cat(unit_shifts_list, dim=0),
        "cell": torch.stack(cells, dim=0),  # [batch, 3, 3]
        "batch": torch.cat(batch_idx, dim=0),
        "ptr": torch.tensor(ptr, dtype=torch.long),
    }
    y = torch.stack(list(targets), dim=0)
    return batched, y

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import h5py
import numpy as np

from ..core import CellType


if TYPE_CHECKING:
    import os
    from typing import Literal
    
    from numpy.typing import NDArray

    from .. import Mesh


def read(filename: str | os.PathLike) -> Mesh:
    """
    Read a MED file.

    Parameters
    ----------
    filename : str | PathLike
        Input file name.

    Returns
    -------
    maillages.Mesh
        Output mesh.
    
    """
    from .. import Mesh

    metadata = {}

    with h5py.File(filename, "r") as f:
        # Read mesh
        ens_maa = f.get("ENS_MAA")

        if enumerate is None:
            raise ValueError("could not find mesh in file")
        
        ens_maa = cast(h5py.Group, ens_maa)
        mesh_names = list(ens_maa)
        mesh = cast(h5py.Group, ens_maa[mesh_names[0]])
        submesh_names = list(mesh)
        submesh = cast(h5py.Group, mesh[submesh_names[0]])

        # Read profiles
        profiles = {}

        if "PROFILS" in f:
            profils = cast(h5py.Group, f["PROFILS"])

            for k, v in profils.items():
                n = int(v.attrs["NBR"])
                profile = cast(h5py.Group, v["PFL"])
                profiles[k] = get_array(profile, n) - 1

        # Read points
        noeuds = cast(h5py.Group, submesh["NOE"])
        points = get_array(noeuds["COO"], "NBR")
        family_id_node = (
            get_array(noeuds["FAM"], "NBR")
            if "FAM" in noeuds
            else None
        )

        # Read cells
        mailles = cast(h5py.Group, submesh["MAI"])
        cells = {}
        family_id_cell, num_id_cell = [], []
        celltype_data = {"NOE": {}}

        for k, v in mailles.items():
            celltype = _med_to_maillages_celltype[k]
            cells[celltype] = get_array(v["NOD"], "NBR") - 1
            celltype_data[str(k)] = {}

            if cells[celltype].ndim == 1:
                cells[celltype] = cells[celltype][:, None]

            if "FAM" in v:
                family_id_cell.append(get_array(v["FAM"], "NBR"))

            if "NUM" in v:
                num_id_cell.append(get_array(v["NUM"], "NBR"))

        # Read families
        if "FAS" in f:
            fas = cast(h5py.Group, f["FAS"])
            fas = cast(h5py.Group, fas[mesh_names[0]])

            if "NOEUD" in fas:
                fas_noeu = cast(h5py.Group, fas["NOEUD"])
                metadata["med:FamilyIdNodeGroup"] = get_families(fas_noeu)

            if "ELEME" in fas:
                fas_eleme = cast(h5py.Group, fas["ELEME"])
                metadata["med:FamilyIdCellGroup"] = get_families(fas_eleme)

        # Read fields
        cha = f.get("CHA")
        time_steps = None

        if cha is not None:
            cha = cast(h5py.Group, cha)

            for k, v in cha.items():
                field = cast(h5py.Group, v)

                # Get variable names
                prefix = k[8:]
                names = cast(bytes, field.attrs["NOM"])
                nco = cast(int, field.attrs["NCO"])

                if names is None:
                    names = [f"{prefix}[{i + 1}]" for i in range(nco)]

                else:
                    names = names.decode().strip().split()
                    names = [f"{prefix}_{name}" for name in names]

                # Get all time steps
                if time_steps is None:
                    time_steps = [time_step.attrs["PDT"] for time_step in v.values()]

                # Get data for each time step
                for time_step in v.values():
                    for kk, vv in time_step.items():
                        if kk != "NOE":
                            _, celltype_ = kk.split(".")

                        else:
                            celltype_ = "NOE"

                    profile_name = vv.attrs["PFL"].decode()
                    profile_ = vv[profile_name]
                    tmp = get_array(profile_["CO"], nco, order="C")

                    nga = profile_.attrs["NGA"]
                    if nga > 1:
                        tmp = tmp.reshape(nco, nga, -1, order="F")
                        tmp = tmp.mean(axis=1)

                    if profile_name == "MED_NO_PROFILE_INTERNAL":
                        for name, arr in zip(names, tmp):
                            celltype_data[celltype_].setdefault(name, []).append(arr)

                    else:
                        nbr = profile_.attrs["NBR"]
                        mask = profiles[profile_name]

                        for name, arr in zip(names, tmp):
                            arr_ = np.full(nbr, np.nan)
                            arr_[mask] = arr
                            celltype_data[celltype_].setdefault(name, []).append(arr_)

    # Post-process field data
    cell_data_names = []

    for k, v in celltype_data.items():
        for kk, vv in v.items():
            if k != "NOE" and kk not in cell_data_names:
                cell_data_names.append(kk)

            celltype_data[k][kk] = np.atleast_1d(vv[0]) if len(vv) == 1 else np.transpose(vv)

    point_data = celltype_data.pop("NOE")

    # Add missing data for other cell types
    n_time_steps = len(time_steps) if time_steps is not None else 1

    for k, v in celltype_data.items():
        for name in cell_data_names:
            if name not in v:
                celltype = _med_to_maillages_celltype[k]
                celltype_data[k][name] = np.full((len(cells[celltype]), n_time_steps), np.nan).squeeze()

    # Convert cell data dict per cell type to a single dict with all cell types
    cell_data = {
        name: np.concatenate(
            [
                celltype_data[k][name] for k in celltype_data
                if name in celltype_data[k]
            ],
            axis=0,
        )
        for name in cell_data_names
    }

    if family_id_node is not None:
        point_data["FamilyIdNode"] = family_id_node

    if family_id_cell:
        cell_data["FamilyIdCell"] = np.concatenate(family_id_cell)

    if num_id_cell:
        cell_data["NumIdCell"] = np.concatenate(num_id_cell) - 1

    return Mesh(
        points,
        cells,
        point_data=point_data,
        cell_data=cell_data,
        time_steps=time_steps,
        metadata=metadata,
    )


def get_array(
    node: h5py.Group | h5py.Dataset | h5py.Datatype,
    n: int | str,
    order: Literal["C", "F"] = "F",
) -> NDArray:
    """Get a NumPy array from an HDF5 node."""
    if isinstance(n, str):
        if node.attrs[n] is None:
            raise ValueError(f"could not find attribute {n} in node {node.name}")

        n = cast(int, node.attrs[n])

    arr = np.asanyarray(node)

    return arr if arr.size == n else arr.reshape(n, -1, order=order)


def get_families(fas: h5py.Group) -> dict:
    """Get the family ID to name mapping from a FAS group."""
    families = {
        node_set.attrs["NUM"]: [
            "".join(map(chr, dataset)).strip().rstrip("\x00")
            for dataset in node_set["GRO"]["NOM"][:]
        ]
        for node_set in fas.values()
    }

    return {
        tuple(v) if len(v) > 1 else v[0]: int(k)
        for k, v in families.items()
    }


_maillages_to_med_celltype = {
    CellType.vertex: "PO1",
    CellType.line: "SE2",
    CellType.line3: "SE3",
    CellType.triangle: "TR3",
    CellType.triangle6: "TR6",
    CellType.quad: "QU4",
    CellType.quad8: "QU8",
    CellType.tetra: "TE4",
    CellType.hexahedron: "HE8",
    CellType.wedge: "PE6",
    CellType.pyramid: "PY5",
}
_med_to_maillages_celltype = {v: k for k, v in _maillages_to_med_celltype.items()}

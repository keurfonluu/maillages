from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pyrequire import require_package


if TYPE_CHECKING:
    from typing import Optional

    import pyvista as pv
    from numpy.typing import ArrayLike, NDArray


class Mesh:
    def __init__(
        self,
        *args,
        point_data: Optional[dict] = None,
        cell_data: Optional[dict] = None,
        point_sets: Optional[dict] = None,
        cell_sets: Optional[dict] = None,
        time_steps: Optional[ArrayLike] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        from .. import CellType

        # Parse input arguments
        if len(args) == 2:
            points, cell_dict = args

            if not isinstance(cell_dict, dict):
                raise ValueError("could not parse cells without cell types")

            cells, celltypes = [], []
            
            for k, v in cell_dict.items():
                cells += list(np.atleast_2d(v))
                celltype = CellType[k] if isinstance(k, str) else k
                celltypes += [int(celltype)] * len(v)

        else:
            raise NotImplementedError()
        
        # Points
        points = np.asanyarray(points)

        if points.shape[1] == 2:
            points = np.insert(points, 2, 0.0, axis=1)

        # Point and cell data
        point_data = point_data if point_data is not None else {}
        point_data = {k: np.asanyarray(v) for k, v in point_data.items()}

        cell_data = cell_data if cell_data is not None else {}
        cell_data = {k: np.asanyarray(v) for k, v in cell_data.items()}

        # Point and cell sets
        point_sets = point_sets if point_sets is not None else {}
        cell_sets = cell_sets if cell_sets is not None else {}

        # Time steps
        if time_steps is not None:
            time_steps = np.asanyarray(time_steps)

            for k, v in point_data.items():
                if v.ndim > 1 and v.shape[-1] != len(time_steps):
                    raise ValueError(f"could not match number of time steps with point data '{k}'")
                
            for k, v in cell_data.items():
                if v.ndim > 1 and v.shape[-1] != len(time_steps):
                    raise ValueError(f"could not match number of time steps with cell data '{k}'")
        
        self._points = points
        self._cells = cells
        self._celltypes = np.array(celltypes)
        self._point_data = point_data
        self._cell_data = cell_data
        self._point_sets = point_sets
        self._cell_sets = cell_sets
        self._time_steps = time_steps
        self._metadata = metadata if metadata is not None else {}

    @require_package("pyvista")
    def to_pyvista(self) -> pv.UnstructuredGrid:
        from ..utils import to_pyvista

        return to_pyvista(self)

    @property
    def cell_data(self) -> dict:
        return self._cell_data
    
    @property
    def cell_sets(self) -> dict:
        return self._cell_sets

    @property
    def cells(self) -> list[NDArray]:
        return self._cells
    
    @property
    def celltypes(self) -> NDArray:
        return self._celltypes
    
    @property
    def metadata(self) -> dict:
        return self._metadata
    
    @property
    def n_cells(self) -> int:
        return len(self.cells)
    
    @property
    def n_points(self) -> int:
        return len(self.points)
    
    @property
    def points(self) -> NDArray:
        return self._points
    
    @property
    def point_data(self) -> dict:
        return self._point_data
    
    @property
    def point_sets(self) -> dict:
        return self._point_sets
    
    @property
    def time_steps(self) -> NDArray | None:
        return self._time_steps

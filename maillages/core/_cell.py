from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pyrequire import require_package


if TYPE_CHECKING:
    from typing import Optional

    import pyvista as pv
    from numpy.typing import ArrayLike, NDArray

    from ._celltype import CellType


class Cell:
    """
    Single cell of a mesh.

    Parameters
    ----------
    points : ArrayLike
        Array of shape (N, 3) containing the coordinates of the cell's vertices.
    celltype : CellType
        Type of the cell.
    point_data : dict, optional
        Dictionary containing data arrays defined at each vertex.
    cell_data : dict, optional
        Dictionary containing scalar data arrays defined for the cell.
    time_steps : ArrayLike, optional
        Array of time steps for time-dependent data. If provided, point and cell data
        arrays must have a last dimension matching the length of time_steps.
    metadata : dict, optional
        Dictionary containing metadata for the cell.

    """

    def __init__(
        self,
        points: ArrayLike,
        celltype: CellType,
        point_data: Optional[dict] = None,
        cell_data: Optional[dict] = None,
        time_steps: Optional[ArrayLike] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Initialize a cell object."""
        points = np.asanyarray(points)

        if points.ndim != 2 or points.shape[1] not in {2, 3}:
            raise ValueError("points must be an array of shape (N, 2) or (N, 3)")

        if points.shape[1] == 2:
            points = np.insert(points, 2, 0.0, axis=1)

        self._points = points
        self._celltype = celltype
        self._point_data = {k: v for k, v in (point_data or {}).items()}
        self._cell_data = {k: v for k, v in (cell_data or {}).items()}
        self._time_steps = np.asanyarray(time_steps) if time_steps is not None else None
        self._metadata = metadata or {}

    @require_package("pyvista")
    def to_pyvista(self) -> pv.Cell:
        """
        Convert the cell to a PyVista cell.

        Returns
        -------
        pyvista.Cell
            Output PyVista cell.

        """
        from ..utils import to_pyvista

        return to_pyvista(self)
        
    @property
    def cell_data(self) -> dict:
        """Get the cell data dictionary."""
        return self._cell_data

    @property
    def celltype(self) -> CellType:
        """Get the cell type."""
        return self._celltype

    @property
    def metadata(self) -> dict:
        """Get the metadata dictionary."""
        return self._metadata

    @property
    def n_points(self) -> int:
        """Get the number of vertices in the cell."""
        return len(self._points)

    @property
    def point_data(self) -> dict:
        """Get the point data dictionary."""
        return self._point_data

    @property
    def points(self) -> NDArray:
        """Get the array of vertex coordinates."""
        return self._points
    
    @property
    def time_steps(self) -> NDArray | None:
        """Get the array of time steps for time-dependent data."""
        return self._time_steps

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from pyrequire import require_package


if TYPE_CHECKING:
    from typing import Literal, Optional

    import pyvista as pv
    from matplotlib.axes import Axes
    from matplotlib.collections import Collection
    from numpy.typing import ArrayLike, NDArray


class Mesh:
    """
    Core mesh class.

    Parameters
    ----------
    *args : tuple[ArrayLike, dict]]
        Positional arguments. Either of the following:

        - points, cell_dict: where points is an (N, 2) or (N, 3) array of point
        coordinates and cell_dict is a dictionary mapping cell types to arrays of cell
        connectivity. Cell types can be specified as either strings or integers.
        - points, cells, celltypes: where points is an (N, 2) or (N, 3) array of point
        coordinates, cells is a list of arrays of cell connectivity, and celltypes is
        an array of integers specifying the cell type for each cell.

    point_data : dict, optional
        Dictionary containing point data arrays.
    cell_data : dict, optional
        Dictionary containing cell data arrays.
    point_sets : dict, optional
        Dictionary containing point sets.
    cell_sets : dict, optional
        Dictionary containing cell sets.
    time_steps : ArrayLike, optional
        Array of time steps for time-dependent data. If provided, point and cell data
        arrays must have a last dimension matching the length of time_steps.
    metadata : dict, optional
        Dictionary containing metadata information.

    """

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
        """Initialize a mesh object."""
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

        elif len(args) == 3:
            points, cells, celltypes = args

        else:
            raise ValueError("invalid number of positional arguments")

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
                    raise ValueError(
                        f"could not match number of time steps with point data '{k}'"
                    )

            for k, v in cell_data.items():
                if v.ndim > 1 and v.shape[-1] != len(time_steps):
                    raise ValueError(
                        f"could not match number of time steps with cell data '{k}'"
                    )

        self._points = points
        self._cells = cells
        self._celltypes = np.array(celltypes)
        self._point_data = point_data
        self._cell_data = cell_data
        self._point_sets = point_sets
        self._cell_sets = cell_sets
        self._time_steps = time_steps
        self._metadata = metadata if metadata is not None else {}

    @require_package("matplotlib")
    def plot(
        self,
        c: Optional[str | ArrayLike] = None,
        cmap: str = "viridis",
        edgecolor: Optional[str | tuple[float, ...]] = None,
        linewidth: float = 0.5,
        axis: int = 2,
        ax: Optional[Axes] = None,
        **kwargs
    ) -> Collection:
        """
        Create a 2D pseudocolor plot of an unstructured grid.

        Parameters
        ----------
        c : str | ArrayLike, optional
            Data array name or values to use for coloring.
        cmap : str, default 'viridis'
            Colormap.
        edgecolor : str | tuple[float, ...], optional
            Color of the wireframe edges. If None, no edges will be drawn.
        linewidth : float, default 0.5
            Width of the wireframe edges.
        axis : {0, 1, 2}, default 2
            Axis to project the points onto for 2D plotting.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, use current axes.
        **kwargs : dict
            Additional keyword arguments. See ``matplotlib.tri.Triangulation`` and
            ``matplotlib.collections.PolyCollection`` for more details.
        
        Returns
        -------
        matplotlib.collections.Collection
            The Collection object created by the plot.

        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        from matplotlib.collections import PolyCollection
        from .. import CellType

        ax = ax if ax is not None else plt.gca()
        points = self.points[:, np.delete(np.arange(3), axis)]

        if not np.isin(self.celltypes, [CellType.triangle, CellType.quad, CellType.polygon]).all():
            raise NotImplementedError

        # Determine data type and values for coloring
        values = None
        is_point_data = False

        if c is not None:
            if isinstance(c, str):
                if c in self.point_data:
                    values = np.asanyarray(self.point_data[c], dtype=float)
                    is_point_data = True

                elif c in self.cell_data:
                    values = np.asanyarray(self.cell_data[c], dtype=float)
                    is_point_data = False

                else:
                    raise ValueError(f"could not find data array named '{c}' in point or cell data")

            else:
                values = np.asanyarray(c, dtype=float)

                if len(values) == len(self.points):
                    is_point_data = True

                elif len(values) == len(self.cells):
                    is_point_data = False

                else:
                    raise ValueError(f"could not determine data type from provided values with length {len(values)}")

        # Plot point data
        if is_point_data:
            triangles = [
                [cell[0], cell[i], cell[i + 1]]
                for cell in self.cells
                for i in range(1, len(cell) - 1)
            ]       
            tri = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
            
            if values is not None and not np.isfinite(values).all():
                invalid_nodes = ~np.isfinite(values)
                mask = np.any(invalid_nodes[tri.triangles], axis=1)
                tri.set_mask(mask)
                safe_values = np.copy(values)
                safe_values[invalid_nodes] = np.nanmean(values) if not np.isnan(values).all() else 0.0

            else:
                safe_values = values

            levels = kwargs.pop("levels", 11)
            safe_values = cast(NDArray, safe_values)
            collection = ax.tricontourf(tri, safe_values, levels=levels, cmap=cmap, **kwargs)
            
            if edgecolor is not None:
                wireframe_col = PolyCollection(
                    [points[cell] for cell in self.cells],
                    facecolors="none",
                    edgecolors=edgecolor,
                    linewidths=linewidth,
                    antialiased=True
                )
                ax.add_collection(wireframe_col)

        # Plot cell data
        else:
            collection = PolyCollection(
                [points[cell] for cell in self.cells],
                edgecolors=edgecolor,
                linewidths=linewidth,
                cmap=cmap,
                antialiased=edgecolor is not None,
                **kwargs
            )
            
            if values is not None:
                collection.set_array(values)
            
            ax.add_collection(collection)

        # Set axis limits and aspect ratio            
        ax.set_xlim(points[:, 0].min(), points[:, 0].max())
        ax.set_ylim(points[:, 1].min(), points[:, 1].max())
        ax.set_aspect("equal")

        return collection

    @require_package("pyvista")
    def to_pyvista(self) -> pv.UnstructuredGrid:
        """
        Convert the mesh to a PyVista grid.

        Returns
        -------
        pyvista.UnstructuredGrid
            Output PyVista grid.

        """
        from ..utils import to_pyvista

        return to_pyvista(self)

    def _get_entity_tags(
        self, entity: Literal["point", "cell"]
    ) -> tuple[NDArray | None, dict | None]:
        """
        Get the integer tags for the specified entity type.

        Parameters
        ----------
        entity : {'point', 'cell'}
            Entity for which to retrieve tags.

        Returns
        -------
        NDArray | None
            Entity tags.
        dict | None
            Mapping of tag values to tag names.

        """
        data = self.point_data if entity == "point" else self.cell_data
        integer_data_keys = [k for k, v in data.items() if v.dtype.kind == "i"]

        for key in integer_data_keys:
            if key in self.metadata:
                tag_to_id = self.metadata[key]
                id_to_tag = {v: k for k, v in tag_to_id.items()}

                return np.array(list(map(lambda x: id_to_tag[x], data[key]))), tag_to_id

        if integer_data_keys:
            tag_to_id = {str(i): i for i in np.unique(data[integer_data_keys[0]])}

            return data[integer_data_keys[0]], tag_to_id

        return None, None

    @property
    def cell_data(self) -> dict:
        """Get the cell data dictionary."""
        return self._cell_data

    @property
    def cell_sets(self) -> dict:
        """Get the cell sets dictionary."""
        return self._cell_sets

    @property
    def cell_tags(self) -> NDArray | None:
        """Get the cell tags array."""
        return self._get_entity_tags("cell")[0]

    @property
    def cells(self) -> list[NDArray]:
        """Get the list of cell connectivity arrays."""
        return self._cells

    @property
    def celltypes(self) -> NDArray:
        """Get the array of cell types."""
        return self._celltypes

    @property
    def metadata(self) -> dict:
        """Get the metadata dictionary."""
        return self._metadata

    @property
    def n_cells(self) -> int:
        """Get the total number of cells in the mesh."""
        return len(self.cells)

    @property
    def n_points(self) -> int:
        """Get the total number of points in the mesh."""
        return len(self.points)

    @property
    def points(self) -> NDArray:
        """Get the array of points."""
        return self._points

    @property
    def point_data(self) -> dict:
        """Get the point data dictionary."""
        return self._point_data

    @property
    def point_sets(self) -> dict:
        """Get the point sets dictionary."""
        return self._point_sets

    @property
    def point_tags(self) -> NDArray | None:
        """Get the point tags array."""
        return self._get_entity_tags("point")[0]

    @property
    def time_steps(self) -> NDArray | None:
        """Get the array of time steps."""
        return self._time_steps

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray
from pyrequire import require_package


if TYPE_CHECKING:
    from typing import Literal, Optional

    import pyvista as pv
    from matplotlib.axes import Axes
    from matplotlib.collections import Collection
    from matplotlib.colors import Colormap
    from matplotlib.tri import TriContourSet
    from numpy.typing import ArrayLike


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

    def __call__(self, t: ArrayLike, eps: float = 1.0e-8) -> Mesh:
        """
        Interpolate the mesh data to the specified time step(s).

        Parameters
        ----------
        t : ArrayLike
            Time step(s) to interpolate to. Must be within the range of time steps
            specified in the mesh.
        eps : float, default 1.0e-8
            Tolerance for determining if a time step matches an existing time step.

        Returns
        -------
        maillages.Mesh
            New mesh object with data interpolated to the specified time step(s).

        """
        t = np.atleast_1d(t)
        t = np.sort(t) if t.ndim == 1 else t

        if self.time_steps is None or len(self.time_steps) < 2:
            raise ValueError("could not interpolate mesh without at least 2 time steps")

        if t[0] < self.time_steps[0] or t[-1] > self.time_steps[-1]:
            raise ValueError(
                f"could not interpolate mesh outside of time step range ({self.time_steps[0]}, {self.time_steps[-1]})"
            )

        # Find the indices of the time steps for interpolation
        ids = np.searchsorted(self.time_steps, t, side="right") - 1

        # Ensure arrays are at least 2D
        n_time_steps = self.n_time_steps
        point_data_2d, cell_data_2d = {}, {}

        for k, v in self.point_data.items():
            point_data_2d[k] = (
                np.repeat(v[:, np.newaxis], n_time_steps, axis=1)
                if v.ndim == 1
                else np.atleast_2d(v)
            )

        for k, v in self.cell_data.items():
            cell_data_2d[k] = (
                np.repeat(v[:, np.newaxis], n_time_steps, axis=1)
                if v.ndim == 1
                else np.atleast_2d(v)
            )

        # Interpolate point and cell data to the specified time steps
        point_data, cell_data = {}, {}

        for t_, id_ in zip(t, ids):
            if np.isclose(t_, self.time_steps[id_], atol=eps):
                for k, v in point_data_2d.items():
                    point_data.setdefault(k, []).append(v[..., id_])

                for k, v in cell_data_2d.items():
                    cell_data.setdefault(k, []).append(v[..., id_])

            else:
                t1, t2 = self.time_steps[id_], self.time_steps[id_ + 1]
                dt = t2 - t1
                w1, w2 = (t2 - t_) / dt, (t_ - t1) / dt

                for k, v in point_data_2d.items():
                    if v.dtype.kind == "i":
                        point_data.setdefault(k, []).append(v[..., id_])

                    else:
                        v1, v2 = v[..., id_], v[..., id_ + 1]
                        point_data.setdefault(k, []).append(w1 * v1 + w2 * v2)

                for k, v in cell_data_2d.items():
                    if v.dtype.kind == "i":
                        cell_data.setdefault(k, []).append(v[..., id_])

                    else:
                        v1, v2 = v[..., id_], v[..., id_ + 1]
                        cell_data.setdefault(k, []).append(w1 * v1 + w2 * v2)

        return Mesh(
            self.points,
            self.cells,
            self.celltypes,
            point_data={
                k: np.stack(v, axis=-1).squeeze() for k, v in point_data.items()
            },
            cell_data={k: np.stack(v, axis=-1).squeeze() for k, v in cell_data.items()},
            time_steps=t,
            metadata=self.metadata,
        )

    def __getitem__(self, key: int | ArrayLike | slice) -> Mesh:
        """
        Select a subset of the mesh based on cell indices.

        Parameters
        ----------
        key : int | ArrayLike | slice
            Indices of the cells to select.

        Returns
        -------
        maillages.Mesh
            New mesh object containing only the selected cells and associated data.

        """
        # Mask for selecting cells
        cell_mask = np.zeros(self.n_cells, dtype=bool)
        cell_mask[key] = True

        # Select cells and cell types
        cells = [cell for cell, mask_ in zip(self.cells, cell_mask) if mask_]
        celltypes = self.celltypes[cell_mask]

        # Select points and remap point indices
        point_mask = np.zeros(self.n_points, dtype=bool)

        for cell in cells:
            point_mask[cell] = True

        points = self.points[point_mask]
        point_index_map = np.full(self.n_points, -1, dtype=int)
        point_index_map[point_mask] = np.arange(point_mask.sum())
        cells = [point_index_map[cell] for cell in cells]

        # Select point and cell data
        point_data = {k: v[point_mask] for k, v in self.point_data.items()}
        cell_data = {k: v[cell_mask] for k, v in self.cell_data.items()}

        return Mesh(
            points,
            cells,
            celltypes,
            point_data=point_data,
            cell_data=cell_data,
            time_steps=self.time_steps,
            metadata=self.metadata,
        )

    @require_package("scipy")
    @require_package("matplotlib")
    def plot(
        self,
        c: Optional[str | ArrayLike] = None,
        cmap: str | Colormap = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        log: bool = False,
        edgecolor: Optional[str | tuple[float, ...]] = None,
        linewidth: float = 0.5,
        fill: bool = True,
        sigma: float = 0.0,
        axis: int = 2,
        component: Optional[int] = None,
        ax: Optional[Axes] = None,
        **kwargs,
    ) -> Collection | TriContourSet:
        """
        Create a 2D pseudocolor plot of an unstructured grid.

        Parameters
        ----------
        c : str | ArrayLike, optional
            Data array name or values to use for coloring.
        cmap : str | Colormap, default 'viridis'
            Colormap to use for coloring.
        vmin : float, optional
            Minimum data value for colormap normalization.
        vmax : float, optional
            Maximum data value for colormap normalization.
        log : bool, default False
            If True, use logarithmic scaling for the colormap.
        edgecolor : str | tuple[float, ...], optional
            Color of the wireframe edges. If None, no edges will be drawn. Ignored if
            fill is False.
        linewidth : float, default 0.5
            Width of the wireframe edges or contour lines.
        fill : bool, default True
            If True, fill the contours. If False, only draw the contour lines. Ignored
            for cell data.
        sigma : float, default 0.0
            Standard deviation for Gaussian kernel for smoothing of contour lines.
        axis : {0, 1, 2}, default 2
            Axis to project the points onto for 2D plotting.
        component : int, optional
            Component of the data array to plot if it has multiple components.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, use current axes.
        **kwargs : dict
            Additional keyword arguments. See ``matplotlib.tri.Triangulation`` and
            ``matplotlib.collections.PolyCollection`` for more details.

        Returns
        -------
        matplotlib.collections.Collection | matplotlib.tri.TriContourSet
            The collection or contour set created by the plot.

        AI Disclosure
        -------------
        The boilerplate of this function was written with the assistance of an AI
        (Google Gemini 3.1 Pro). The code was subsequently reviewed, verified, and
        tested by the maintainer.

        Synthesized prompt used:
        "Write a Python method to plot a 2D unstructured grid of mixed polygons
        (triangles, quads, arbitrary polygons). The method must take an optional data
        input and automatically infer if it represents cell data or point data based on
        its length. For cell data, render the mesh using a flat-shaded PolyCollection.
        For point data, decompose the polygons into a triangle fan and render it using
        VTK-style filled contours (tricontourf). Implement safety masking to handle
        NaN/inf values without crashing the triangulation engine."

        """
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        import numpy as np
        from matplotlib.collections import PolyCollection

        from .. import CellType

        ax = ax if ax is not None else plt.gca()
        points = self.points[:, np.delete(np.arange(3), axis)]

        if not np.isin(
            self.celltypes, [CellType.triangle, CellType.quad, CellType.polygon]
        ).all():
            raise NotImplementedError

        # Additional keyword arguments for contouring
        levels = kwargs.pop("levels", 11)
        colors = kwargs.pop("colors", None)

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
                    raise ValueError(
                        f"could not find data array named '{c}' in point or cell data"
                    )

            else:
                values = np.asanyarray(c, dtype=float)

                if len(values) == len(self.points):
                    is_point_data = True

                elif len(values) == len(self.cells):
                    is_point_data = False

                else:
                    raise ValueError(
                        f"could not determine data type from provided values with length {len(values)}"
                    )

        # Handle component selection for multi-component data
        if values is not None and values.ndim > 1:
            component = component if component is not None else -1
            values = values[..., component]

            if values.ndim == 2:
                values = np.linalg.norm(values, axis=-1)

            elif values.ndim > 2:
                raise ValueError(f"could not plot data with more than 3 dimensions")

        # Apply spatial Gaussian smoothing
        if values is not None and is_point_data and sigma > 0.0:
            values = self._gaussian_filter(points, values, sigma)

        # Set colormap normalization limits
        if values is not None:
            vmin = np.nanmin(values) if vmin is None else vmin
            vmax = np.nanmax(values) if vmax is None else vmax

        # Set log scale
        if log:
            if values is not None:
                mask = values > 0.0
                values = np.copy(values)
                values[mask] = np.log10(values[mask])

                if vmin is not None:
                    vmin = np.log10(vmin) if vmin > 0.0 else None
                    values[~mask] = vmin

                else:
                    values[~mask] = np.nan

                if vmax is not None:
                    vmax = np.log10(vmax) if vmax > 0.0 else None

            if isinstance(levels, Sequence):
                levels = np.array(levels)
                levels = np.log10(levels[levels > 0.0])

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
                safe_values[invalid_nodes] = (
                    np.nanmean(values) if not np.isnan(values).all() else 0.0
                )

            else:
                safe_values = values

            safe_values = cast(NDArray, safe_values)
            contour = ax.tricontourf if fill else ax.tricontour
            collection = contour(
                tri,
                safe_values,
                levels=levels,
                colors=colors,
                cmap=cmap if colors is None else None,
                vmin=vmin,
                vmax=vmax,
                **kwargs,
            )

            if fill and edgecolor is not None:
                wireframe = PolyCollection(
                    [points[cell] for cell in self.cells],
                    facecolors="none",
                    edgecolors=edgecolor,
                    linewidths=linewidth,
                    antialiased=True,
                )
                ax.add_collection(wireframe)

        # Plot cell data
        else:
            collection = PolyCollection(
                [points[cell] for cell in self.cells],
                edgecolors=edgecolor,
                linewidths=linewidth,
                cmap=cmap,
                antialiased=edgecolor is not None,
                **kwargs,
            )

            if values is not None:
                collection.set_array(values)
                collection.set_clim(vmin, vmax)

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

    @staticmethod
    @require_package("scipy")
    def _gaussian_filter(points: NDArray, values: NDArray, sigma: float) -> NDArray:
        """
        Apply a spatial Gaussian filter to unstructured points.

        AI Disclosure
        -------------
        The boilerplate of this function was written with the assistance of an AI
        (Google Gemini 3.1 Pro). The code was subsequently reviewed, verified, and
        tested by the maintainer.

        Synthesized prompt used:
        "Implement a spatial Gaussian filter using a KD-Tree to smooth the unstructured
        data."

        """
        import numpy as np
        from scipy.spatial import KDTree

        tree = KDTree(points)
        smoothed_values = np.empty_like(values)
        radius = 3.0 * sigma

        for i, point in enumerate(points):
            ids = tree.query_ball_point(point, r=radius)
            neighbor_vals = values[ids]
            neighbor_points = points[ids]

            # Calculate squared distances from the target point
            d2 = np.sum((neighbor_points - point) ** 2, axis=1)
            valid = np.isfinite(neighbor_vals)

            if np.sum(valid) == 0:
                smoothed_values[i] = np.nan
                continue

            # Apply Gaussian weight function and calculate weighted average
            weights = np.exp(-d2[valid] / (2 * sigma**2))
            smoothed_values[i] = np.sum(weights * neighbor_vals[valid]) / np.sum(
                weights
            )

        return smoothed_values

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
    def cell_centers(self) -> NDArray:
        """Get the array of cell centers."""
        from .. import CellType

        centers = [
            self.points[cell].mean(axis=0)
            if celltype != CellType.empty
            else [np.nan, np.nan, np.nan]
            for cell, celltype in zip(self.cells, self.celltypes)
        ]

        return np.array(centers, dtype=float)

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
    def n_time_steps(self) -> int:
        """Get the number of time steps in the mesh."""
        return len(self.time_steps) if self.time_steps is not None else 1

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

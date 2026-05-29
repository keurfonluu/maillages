from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pyrequire import require_package


if TYPE_CHECKING:
    import os

    from .. import Mesh


@require_package("shapefile")
def read(filename: str | os.PathLike) -> Mesh:
    """
    Read a shapefile.

    Parameters
    ----------
    filename : str | PathLike
        Input file name.

    Returns
    -------
    maillages.Mesh
        Output mesh.

    """
    import shapefile

    from .. import CellType, Mesh

    with shapefile.Reader(filename) as shp:
        points, cells, celltypes, cell_data = [], [], [], {}
        polygon_holes = []

        for i, shape_record in enumerate(shp.iterShapeRecords()):
            shape, record = shape_record.shape, shape_record.record

            if shape is None or record is None:
                continue

            shape_type = shape.shapeTypeName
            shape_z = shape.z if hasattr(shape, "z") else np.zeros(len(shape.points))
            shape_points = [(x, y, float(z)) for (x, y), z in zip(shape.points, shape_z)]
            shape_cell = [list(range(len(points), len(points) + len(shape_points)))]
            shape_data = {k: [v] for k, v in record.as_dict().items()}

            if shape_type.startswith("POINT"):
                shape_celltype = [CellType.vertex] * len(shape_points)
                polygon_holes += [-1] * len(shape_points)

            elif shape_type.startswith("POLYGON"):
                # No hole
                if shape_points[0] == shape_points[-1]:
                    shape_points = shape_points[:-1]
                    shape_cell = [list(range(len(points), len(points) + len(shape_points)))]
                    shape_celltype = [CellType.polygon]
                    polygon_holes.append(-1)

                # With hole(s)
                else:
                    is_hole, polygon = False, []
                    shape_points_, shape_cell, shape_celltype = [], [], []
                    n_points = len(points)

                    for ip, point in enumerate(shape_points):
                        polygon.append(point)

                        if (
                            len(polygon) > 1
                            and ip < len(shape_points) - 1
                            and point == polygon[0]
                        ):
                            shape_points_ += polygon[:-1]
                            shape_cell.append(list(range(n_points, n_points + len(polygon) - 1)))
                            shape_celltype.append(CellType.polygon)
                            polygon_holes.append(i if is_hole else -1)
                            n_points += len(polygon) - 1
                            is_hole, polygon = True, []

                    shape_points = shape_points_
                    shape_data = {k: v * len(shape_cell) for k, v in shape_data.items()}

            elif shape_type.startswith("POLYLINE"):
                shape_celltype = [CellType.line]
                polygon_holes.append(-1)

            else:
                raise NotImplementedError(
                    f"shape type '{shape_type}' are not supported yet"
                )
            
            points += shape_points
            cells += shape_cell
            celltypes += shape_celltype

            for k, v in shape_data.items():
                cell_data.setdefault(k, []).extend(v)

        if max(polygon_holes) >= 0:
            cell_data["PolygonHole"] = polygon_holes

    return Mesh(points, cells, celltypes, cell_data=cell_data)

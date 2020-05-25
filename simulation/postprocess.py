from pathlib import Path
from typing import Dict, Tuple

import numpy as np  # type: ignore
from vtk import (  # type: ignore
    vtkPoints,
    vtkPolyData,
    vtkStructuredPoints,
    vtkXMLImageDataWriter,
    vtkXMLPolyDataWriter,
)
from vtk.util.numpy_support import numpy_to_vtk  # type: ignore

from simulation.cell import CellList
from simulation.grid import RectangularGrid
from simulation.modules.geometry import GeometryState
from simulation.state import State


def convert_cells(cells: CellList) -> vtkPolyData:
    cell_data = cells.cell_data
    fields = dict(cell_data.dtype.fields)
    fields.pop('point')

    points = vtkPoints()
    poly = vtkPolyData()
    poly.SetPoints(points)

    if not len(cell_data):
        return poly

    # vtk uses coordinate ordering x, y, z while we use z, y, x.
    points.SetData(numpy_to_vtk(np.flip(cell_data['point'], axis=1)))
    point_data = poly.GetPointData()

    for field, (dtype, _) in fields.items():
        data = cell_data[field]

        # numpy_to_vtk doesn't handle bool for some reason
        if dtype == np.dtype('bool'):
            data = data.astype(np.dtype('uint8'))

        try:
            scalar = numpy_to_vtk(data)
        except Exception:
            print(f'Unhandled data type in field {field}')
            continue

        scalar.SetName(field)
        point_data.AddArray(scalar)

    return poly


def create_volume(grid: RectangularGrid, geometry: GeometryState) -> vtkStructuredPoints:
    x = grid.x
    y = grid.y
    z = grid.z
    vtk_grid = vtkStructuredPoints()
    vtk_grid.SetDimensions(len(x), len(y), len(z))

    # In theory, the rectangular grid used in our code is more general than
    # than a vtkStructuredPoints object.  In practice, we always construct
    # a uniform grid, so we choose to use the more efficient vtk data structure.
    vtk_grid.SetOrigin(0, 0, 0)
    vtk_grid.SetSpacing(x[1] - x[0], y[1] - y[0], z[1] - z[0])

    point_data = vtk_grid.GetPointData()
    point_data.SetScalars(numpy_to_vtk(geometry.lung_tissue.ravel()))
    return vtk_grid


def generate_vtk_objects(state: State) -> Tuple[vtkStructuredPoints, Dict[str, vtkPolyData]]:
    volume = create_volume(state.grid, state.geometry)
    cells = {
        'spore': convert_cells(state.fungus.cells),
        'epithelium': convert_cells(state.epithelium.cells),
        'macrophage': convert_cells(state.macrophage.cells),
        'neutrophil': convert_cells(state.neutrophil.cells),
    }

    return volume, cells


def process_output(file_name: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    state = State.load(file_name)
    volume, cells = generate_vtk_objects(state)

    grid_writer = vtkXMLImageDataWriter()
    grid_writer.SetDataModeToBinary()
    grid_writer.SetFileName(str(output_dir / 'geometry_001.vti'))
    grid_writer.SetInputData(volume)
    grid_writer.Write()

    cell_writer = vtkXMLPolyDataWriter()
    cell_writer.SetDataModeToBinary()
    for module, data in cells.items():
        cell_writer.SetFileName(str(output_dir / f'{module}_001.vtp'))
        cell_writer.SetInputData(data)
        cell_writer.Write()

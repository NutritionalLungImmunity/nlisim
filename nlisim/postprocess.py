from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

# Import from vtkmodules, instead of vtk, to avoid requiring OpenGL
import numpy as np  # type: ignore
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkStructuredPoints
from vtkmodules.vtkIOXML import vtkXMLImageDataWriter, vtkXMLPolyDataWriter

from nlisim.cell import CellData, CellList
from nlisim.grid import RectangularGrid
from nlisim.state import State


def convert_cells_to_vtk(cells: CellList) -> vtkPolyData:
    cell_data: CellData = cells.cell_data
    live_cells = cells.alive()
    cell_data = cell_data[live_cells]

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

    for field, (dtype, *_) in fields.items():
        data = cell_data[field]

        # numpy_to_vtk doesn't handle bool for some reason
        if dtype == np.dtype('bool'):
            data = data.astype(np.dtype('uint8'))

        # noinspection PyBroadException
        try:
            scalar = numpy_to_vtk(data)
        except Exception:
            print(f'Unhandled data type in field {field}')
            continue

        scalar.SetName(field)
        point_data.AddArray(scalar)

    return poly


def create_vtk_volume(grid: RectangularGrid) -> vtkStructuredPoints:
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
    return vtk_grid


def create_vtk_geometry(grid: RectangularGrid, lung_tissue: np.ndarray) -> vtkStructuredPoints:
    vtk_grid = create_vtk_volume(grid)
    point_data = vtk_grid.GetPointData()

    # transform color values to get around categorical interpolation issue in visualization
    tissue = lung_tissue.copy()  # copy required as postprocessing can be live
    tissue[tissue == 0] = 4
    point_data.SetScalars(numpy_to_vtk(tissue.ravel()))
    return vtk_grid


# def create_vtk_molecules(grid: RectangularGrid, molecules: MoleculesState) -> vtkStructuredPoints:
#     vtk_grid = create_vtk_volume(grid)
#     point_data = vtk_grid.GetPointData()
#     for name in molecules.grid.concentrations.dtype.names:
#         data = numpy_to_vtk(molecules.grid.concentrations[name].ravel())
#         data.SetName(name)
#         point_data.AddArray(data)
#
#     return vtk_grid


def add_vtk_molecules(
    molecules: np.ndarray, module_name: str, vtk_grid: vtkStructuredPoints
) -> vtkStructuredPoints:
    point_data = vtk_grid.GetPointData()
    # the x-ferrin molecules have a record type which has several names, others do not
    if molecules.dtype.names:
        for name in molecules.dtype.names:
            data = numpy_to_vtk(molecules[name].ravel())
            data.SetName(name)  # TODO: should we put the module name as part?
            point_data.AddArray(data)
    else:
        data = numpy_to_vtk(molecules.ravel())
        data.SetName(module_name)
        point_data.AddArray(data)

    return vtk_grid


def generate_vtk_objects(
    state: State,
) -> Tuple[vtkStructuredPoints, vtkStructuredPoints, Dict[str, vtkPolyData]]:
    """Generate the vtk objects for each module. (e.g. for upload)"""
    volume = create_vtk_geometry(state.grid, state.lung_tissue)
    molecules_grid = create_vtk_volume(state.grid)
    cells = dict()
    for module in state.config.modules:
        data_type, content = module.visualization_data(state)
        if data_type == 'molecule':
            assert isinstance(content, np.ndarray)
            add_vtk_molecules(content, module.name, molecules_grid)
        elif data_type == 'cells':
            assert isinstance(content, CellList)
            cells[module.name] = convert_cells_to_vtk(content)

    return volume, molecules_grid, cells


def generate_vtk(state: State, postprocess_step_dir: Path):
    volume, molecules, cells = generate_vtk_objects(state)

    grid_writer = vtkXMLImageDataWriter()
    grid_writer.SetDataModeToBinary()
    grid_writer.SetFileName(str(postprocess_step_dir / 'geometry_001.vti'))
    grid_writer.SetInputData(volume)
    grid_writer.Write()

    grid_writer.SetFileName(str(postprocess_step_dir / 'molecules_001.vti'))
    grid_writer.SetInputData(molecules)
    grid_writer.Write()

    cell_writer = vtkXMLPolyDataWriter()
    cell_writer.SetDataModeToBinary()
    for module, data in cells.items():
        cell_writer.SetFileName(str(postprocess_step_dir / f'{module}_001.vtp'))
        cell_writer.SetInputData(data)
        cell_writer.Write()


def process_output(state_files: Iterable[Path], postprocess_dir: Path) -> None:
    for state_file_index, state_file in enumerate(sorted(state_files)):
        state = State.load(state_file)

        postprocess_step_dir = postprocess_dir / ('%03i' % (state_file_index + 1))
        postprocess_step_dir.mkdir()
        generate_vtk(state, postprocess_step_dir)


def generate_summary_stats(state: State) -> Dict[str, Dict[str, Any]]:
    """Generate summary statistics for the simulation.

    Polls each loaded module for its summary statistics, producing a nested dictionary
    where the first key is the module name and the second key is the statistic name.
    e.g. stats['molecules']['iron_mean']
    modules reporting no statistics are omitted
    """
    simulation_stats = dict()
    for module in state.config.modules:
        module_stats: Dict[str, Any] = module.summary_stats(state)
        if len(module_stats) > 0:
            simulation_stats[module.name] = module_stats
    return simulation_stats

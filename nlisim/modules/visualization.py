from enum import Enum
import json

# Import from vtkmodules, instead of vtk, to avoid requiring OpenGL
import attr
import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkLine, vtkPolyData, vtkStructuredPoints
from vtkmodules.vtkIOLegacy import vtkPolyDataWriter, vtkStructuredPointsWriter

from nlisim.cell import CellList
from nlisim.module import ModuleModel, ModuleState
from nlisim.modules.afumigatus import AfumigatusCellTreeList
from nlisim.modules.fungus import FungusCellData
from nlisim.modules.geometry import TissueTypes
from nlisim.state import State


class VTKTypes(Enum):
    """a enum class for the vtk data type."""

    STRUCTURED_POINTS = 0
    STRUCTURED_GRID = 1
    RECTILINEAR_GRID = 2
    UNSTRUCTURED_GRID = 3
    POLY_DATA = 4


@attr.s(kw_only=True, repr=False)
class VisualizationState(ModuleState):
    last_visualize: float = attr.ib(default=0)

    def __repr__(self):
        return 'VisualizationState(last_visualize)'


class Visualization(ModuleModel):
    name = 'visualization'

    StateClass = VisualizationState

    @classmethod
    def write_poly_data(cls, var, filename: str, attr_names: str) -> None:
        vol = vtkPolyData()
        verts = vtkPoints()
        lines = vtkCellArray()

        if isinstance(var, AfumigatusCellTreeList):
            adjacency = var.adjacency

            for i, j in adjacency.keys():
                if i != j:
                    line = vtkLine()
                    line.GetPointIds().SetId(0, i)
                    line.GetPointIds().SetId(1, j)
                    lines.InsertNextCell(line)

        elif not isinstance(var, CellList):
            raise NotImplementedError(
                f'Only supported CellTree or CellList for POLY_DATA. \
                Got {type(var)}'
            )

        for index in var.alive():
            cell = var[index]
            verts.InsertNextPoint(cell['point'][2], cell['point'][1], cell['point'][0])

        alive_cells = np.take(var.cell_data, var.alive())
        for attr_name in attr_names:
            attr = alive_cells[attr_name]
            scalars = numpy_to_vtk(num_array=attr)
            scalars.SetName(attr_name)
            vol.GetPointData().AddArray(scalars)

        vol.SetPoints(verts)
        vol.SetLines(lines)
        writer = vtkPolyDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(vol)
        writer.Write()

    @classmethod
    def write_structured_points(
        cls, var: np.ndarray, filename: str, dx: float, dy: float, dz: float
    ) -> None:
        vol = vtkStructuredPoints()

        # set dimensions X, Y, Z
        vol.SetDimensions(var.shape[2], var.shape[1], var.shape[0])
        vol.SetOrigin(0, 0, 0)
        vol.SetSpacing(dx, dy, dz)

        scalars = numpy_to_vtk(num_array=var.ravel())

        vol.GetPointData().SetScalars(scalars)
        writer = vtkStructuredPointsWriter()
        writer.SetFileName(filename)
        writer.SetInputData(vol)
        writer.Write()

    def visualize(self, state: State, json_config: dict, filename: str) -> None:
        module_name = json_config['module']
        var_name = json_config['variable']
        vtk_type = json_config['vtk_type']
        attr_names = json_config['attributes']
        var = getattr(getattr(state, module_name), var_name)

        if vtk_type == VTKTypes.STRUCTURED_POINTS.name:
            spacing = (
                state.config.getfloat('simulation', 'dz'),
                state.config.getfloat('simulation', 'dy'),
                state.config.getfloat('simulation', 'dx'),
            )
            if attr_names:
                for attr_name in attr_names:
                    attr = getattr(var, attr_name)

                    # composite data
                    if attr.dtype.names:
                        for name in attr.dtype.names:
                            file_name = filename.replace('<variable>', module_name + '-' + name)
                            Visualization.write_structured_points(
                                attr[name], file_name, spacing[2], spacing[1], spacing[0]
                            )
                    else:
                        file_name = filename.replace('<variable>', module_name + '-' + attr)
                        Visualization.write_structured_points(
                            attr, file_name, spacing[2], spacing[1], spacing[0]
                        )

            else:
                file_name = filename.replace('<variable>', module_name + '-' + var_name)
                Visualization.write_structured_points(
                    var, file_name, spacing[2], spacing[1], spacing[0]
                )

        elif vtk_type == VTKTypes.POLY_DATA.name:
            file_name = filename.replace('<variable>', module_name + '-' + var_name)
            Visualization.write_poly_data(var, file_name, attr_names)

        elif vtk_type == VTKTypes.STRUCTURED_GRID.name:
            raise NotImplementedError('structred_grid is not supported yet')

        elif vtk_type == VTKTypes.RECTILINEAR_GRID.name:
            raise NotImplementedError('rectilinear_grid is not supported yet')

        elif vtk_type == VTKTypes.UNSTRUCTURED_GRID.name:
            raise NotImplementedError('unstructured_grid is not supported yet')

        else:
            raise TypeError(f'Unknown VTK data type: {vtk_type}')

    def advance(self, state: State, previous_time: float) -> State:
        visualization_file_name = self.config.get('visualization_file_name')
        variables = self.config.get('visual_variables')
        print_to_stdout = self.config.getboolean('print_to_stdout')
        json_config = json.loads(variables)
        now = state.time

        if print_to_stdout:
            print(
                '\t'.join(
                    map(
                        str,
                        [
                            len(state.neutrophil.cells.alive()),
                            len(state.fungus.cells.alive()),
                            len(state.macrophage.cells.alive()),
                            len(
                                state.fungus.cells.alive(
                                    state.fungus.cells.cell_data['form']
                                    == FungusCellData.Form.CONIDIA
                                )
                            ),
                            np.sum(state.molecules.grid['iron']),
                            np.std(state.molecules.grid['iron']),
                            np.mean(state.molecules.grid['iron']),
                        ],
                    )
                ),
                end='\t',
            )

        i_f_tot = 0
        cells = state.fungus.cells
        for i in cells.alive():
            i_f_tot += cells.cell_data[i]['iron']

        mask = np.argwhere(state.geometry.lung_tissue != TissueTypes.BLOOD.value)
        i_level = 0
        for [x, y, z] in mask:
            i_level += state.molecules.grid['iron'][x, y, z]

        if print_to_stdout:
            print(str(i_level) + '\t' + str(i_f_tot))

        for variable in json_config:
            file_name = visualization_file_name.replace('<time>', ('%005.0f' % now).strip())
            self.visualize(state, variable, file_name)
            state.visualization.last_visualize = now

        return state

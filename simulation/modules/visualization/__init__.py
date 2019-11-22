from math import floor
from time import time
from enum import Enum

import attr
import numpy as np
import pylab
import vtk
from vtk.util import numpy_support

from simulation.config import SimulationConfig
from simulation.module import Module, ModuleState
from simulation.state import State
from simulation.cell import CellArray, CellTree
from simulation.coordinates import Point


class VTKTypes(Enum):
    '''a enum class for the vtk data type'''
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

class Visualization(Module):
    name = 'visualization'
    defaults = {
        'visualize_interval': '1',
        'visualization_file_name': 'output/<variable>-<time>.vtk'
    }
    StateClass = VisualizationState

    @classmethod
    def write_poly_data(cls, var, filename: str) -> None:
        vol = vtk.vtkPolyData()
        verts = vtk.vtkPoints()
        lines = vtk.vtkCellArray()

        # Currently, only CellTree is supported. Could add more datatype in the futrue
        if (isinstance(var, CellTree)):
            cells = var.cells # get cell array
            for cell in cells:
                verts.InsertNextPoint(cell['point'][2], cell['point'][1], cell['point'][0])

            adjacency = var.adjacency

            for i, j in adjacency.keys():
                if i != j:
                    line = vtk.vtkLine()
                    line.GetPointIds().SetId(0, i)
                    line.GetPointIds().SetId(1, j)
                    lines.InsertNextCell(line)

        else:
            raise NotImplementedError(f'Only supported CellTree for POLY_DATA. Got {type(var)}')

        vol.SetPoints(verts)
        vol.SetLines(lines)
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(vol)
        writer.Write()

    @classmethod
    def write_structured_points(cls, var: np.ndarray, filename: str,
                                dx: float = 0.1, 
                                dy: float = 0.1, 
                                dz: float = 0.1) -> None:

        vol = vtk.vtkStructuredPoints()

        # set dimensions X, Y, Z
        vol.SetDimensions(var.shape[2], var.shape[1], var.shape[0])
        vol.SetOrigin(0, 0, 0)
        vol.SetSpacing(dx, dy, dz)

        scalars = numpy_support.numpy_to_vtk(num_array = var.ravel())
        #scalars = scalars.reshape(var.shape)

        vol.GetPointData().SetScalars(scalars)
        writer = vtk.vtkStructuredPointsWriter()
        writer.SetFileName(filename)
        writer.SetInputData(vol)
        writer.Write()



    @classmethod
    def visualize(cls, state: State, variable: str, filename: str) -> None:
        variable_name, vtk_type = variable.split('|')
        module_name, var_name = variable_name.split('.')
        file_name = filename.replace('<variable>', module_name + '-' + var_name)
        var = getattr(getattr(state, module_name), var_name)

        if vtk_type == VTKTypes.STRUCTURED_POINTS.name:
            Visualization.write_structured_points(var, file_name)

        elif vtk_type == VTKTypes.POLY_DATA.name:
            Visualization.write_poly_data(var, file_name)

        elif vtk_type == VTKTypes.STRUCTURED_GRID.name:
            raise NotImplementedError('structred_grid is not supported yet')

        elif vtk_type == VTKTypes.RECTILINEAR_GRID.name:
            raise NotImplementedError('rectilinear_grid is not supported yet')

        elif vtk_type == VTKTypes.UNSTRUCTURED_GRID.name:
            raise NotImplementedError('unstructured_grid is not supported yet')

        else:
            raise TypeError(f'Unknown VTK data type: {vtk_type}')

    def advance(self, state: State, previous_time: float) -> State:
        visualize_interval = self.config.getfloat('visualize_interval')
        visualization_file_name = self.config.get('visualization_file_name')
        variables = SimulationConfig.parselist(self.config.get('visual_variables'))
        now = state.time;

        if now - state.visualization.last_visualize > visualize_interval - 1e-8:
            for variable in variables:
                file_name = visualization_file_name.replace('<time>', ('%010.3f' % now).strip())
                self.visualize(state, variable, file_name)
                state.visualization.last_visualize = now

        return state

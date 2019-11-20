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
        return 'VisualizationState(last_draw)'

class Visualization(Module):
    name = 'visualization'
    defaults = {
        'visualize_interval': '1'
    }
    StateClass = VisualizationState

    @classmethod
    def write_structured_points(cls, var: np.ndarray, name: str,
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
        writer.SetFileName(name)
        writer.SetInputData(vol)
        writer.Write()



    @classmethod
    def visualize(cls, state: State, variable: str) -> None:
        variable_name, vtk_type = variable.split('|')
        module_name, var_name = variable_name.split('.')
        var = getattr(getattr(state, module_name), var_name)

        if vtk_type == VTKTypes.STRUCTURED_POINTS.name:
            Visualization.write_structured_points(var, var_name)

        elif vtk_type == VTKTypes.POLY_DATA.name:
            pass

        elif vtk_type == VTKTypes.STRUCTURED_GRID.name:
            raise NotImplementedError('structred_grid is not supported yet')

        elif vtk_type == VTKTypes.RECTILINEAR_GRID.name:
            raise NotImplementedError('rectilinear_grid is not supported yet')

        elif vtk_type == VTKTypes.UNSTRUCTURED_GRID.name:
            raise NotImplementedError('unstructured_grid is not supported yet')

        else:
            raise TypeError(f'Unknown VTK data type: {vtk_type}')

    def advance(self, state: State, previous_time: float) -> State:
        draw_interval = self.config.getfloat('visualize_interval')
        variables = SimulationConfig.parselist(self.config.get('visual_variables'))

        now = time()
        if now - state.visualization.last_visualize >= draw_interval:
            for variable in variables:
                self.visualize(state, variable)
                state.visualization.last_visualize = time()

        return state

import attr
import numpy as np
import h5py
import os
import vtk

from simulation.module import Module, ModuleState
from simulation.modules.advection.differences import gradient, laplacian
from simulation.state import grid_variable, State
from simulation.validation import ValidationError


@attr.s(kw_only=True, repr=False)
class GeometryState(ModuleState):
    geometry_grid = grid_variable(np.dtype('int'))

    @geometry_grid.validator
    def _validate_geometry_grid(self, attribute: attr.Attribute, value: np.ndarray) -> None:
        if not (value >= 0).all():
            raise ValidationError(f'not >= 0')

    def __repr__(self):
        return f'GeometryState(geometry_grid)'


class Geometry(Module):
    name = 'geometry'
    defaults = {
        'geometry_grid': '0'
    }
    StateClass = GeometryState

    def initialize(self, state: State):
        geometry: GeometryState = state.geometry

        #g = np.load('./simulation/modules/geometry/geometry.npy')
        with h5py.File(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'geometry.hdf5'), 'r') as f:
            g = f['geometry'][:]

        if (g.shape != state.grid.shape):
            raise ValidationError(f'imported geometry doesn\'t match the shape of primary grid')

        geometry.geometry_grid = np.copy(g)
        print(geometry.geometry_grid.shape)
        
        self.preview(geometry.geometry_grid)

        return state

    def advance(self, state: State, previous_time: float):
        """Advance the state by a single time step."""
        # do nothing since the geoemtry is constant
        return state

    def preview(self, grid: np.ndarray):
        print(vtk.vtkVersion.GetVTKSourceVersion())

        dataImporter = vtk.vtkImageImport()

        #g = np.frombuffer(self.geo.get_obj())
        xbin = grid.shape[2]
        ybin = grid.shape[1]
        zbin = grid.shape[0]
        g = np.uint8(grid.reshape(grid.shape[0] * grid.shape[1] * grid.shape[2]))

        data_string = g.tostring()
        dataImporter.CopyImportVoidPointer(data_string, len(data_string))
        dataImporter.SetDataScalarTypeToUnsignedChar()
        dataImporter.SetNumberOfScalarComponents(1)

        dataImporter.SetDataExtent(0, xbin - 1, 0, ybin - 1, 0, zbin - 1)
        dataImporter.SetWholeExtent(0, xbin - 1, 0,  ybin - 1, 0, zbin - 1)

        # Create transfer mapping scalar value to opacity
        opacityTransferFunction = vtk.vtkPiecewiseFunction()
        opacityTransferFunction.AddPoint(0, 0.0)
        opacityTransferFunction.AddPoint(1, 0.2)
        opacityTransferFunction.AddPoint(2, 0.005)
        opacityTransferFunction.AddPoint(3, 1)
        opacityTransferFunction.AddPoint(4, 0.2)
        opacityTransferFunction.AddPoint(5, 0.2)


        # Create transfer mapping scalar value to color
        colorTransferFunction = vtk.vtkColorTransferFunction()
        colorTransferFunction.AddRGBPoint(0, 0.0, 0.0, 1.0)
        colorTransferFunction.AddRGBPoint(1, 1.0, 0.0, 0.0)
        colorTransferFunction.AddRGBPoint(2, 0.0, 0.0, 1.0)
        colorTransferFunction.AddRGBPoint(3, 1.0, 1.0, 1.0)
        colorTransferFunction.AddRGBPoint(4, 1.0, 1.0, 1.0)
        colorTransferFunction.AddRGBPoint(5, 1.0, 1.0, 1.0)

        # The property describes how the data will look
        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetColor(colorTransferFunction)
        volumeProperty.SetScalarOpacity(opacityTransferFunction)
        #volumeProperty.ShadeOn()
        volumeProperty.SetInterpolationTypeToLinear()

        # The mapper / ray cast function know how to render the data
        volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
        volumeMapper.SetBlendModeToComposite()
        volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

        # The volume holds the mapper and the property and
        # can be used to position/orient the volume
        volume = vtk.vtkVolume()
        volume.SetMapper(volumeMapper)
        volume.SetProperty(volumeProperty)

        ren = vtk.vtkRenderer()
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)

        ren.AddVolume(volume)
        ren.SetBackground(1, 1, 1)
        renWin.SetSize(600, 600)
        renWin.Render()

        def CheckAbort(obj, event):
            if obj.GetEventPending() != 0:
                obj.SetAbortRender(1)

        renWin.AddObserver("AbortCheckEvent", CheckAbort)

        iren.Initialize()
        renWin.Render()
        iren.Start()
import attr
import numpy as np
import h5py
import os
import vtk
from enum import Enum

from simulation.module import Module, ModuleState
from simulation.state import grid_variable, State
from simulation.validation import ValidationError

# I am not quite sure if we should put the definition of the lung tissue types here
class TissueTypes(Enum):
    AIR = 0
    BLOOD = 1
    OTHER = 2
    EPITHELIUM = 3
    SURFACTANT = 4
    PORE = 5

    @classmethod
    def validate(cls, value: np.ndarray):
        return np.logical_and(value >= 0, value <= 5).all() and np.issubclass_(value.dtype.type, np.integer)
        
@attr.s(kw_only=True, repr=False)
class GeometryState(ModuleState):
    lung_tissue = grid_variable(np.dtype('int'))

    @lung_tissue.validator
    def _validate_lung_tissue(self, attribute: attr.Attribute, value: np.ndarray) -> None:
        if not TissueTypes.validate(value):
            raise ValidationError('input illegal')

    def __repr__(self):
        return 'GeometryState(lung_tissue)'


class Geometry(Module):
    name = 'geometry'
    defaults = {
        'geometry_path': 'geometry.hdf5',
        'preview_geometry': 'False'
    }
    StateClass = GeometryState

    def initialize(self, state: State):
        geometry: GeometryState = state.geometry
        preview_geometry = self.config.getboolean('preview_geometry')
        path = self.config.get('geometry_path')
        try:
            with h5py.File(path, 'r') as f:
                if f['geometry'][:].shape != state.grid.shape:
                    raise ValidationError('imported geometry doesn\'t match the shape of primary grid')
                geometry.lung_tissue[:] = f['geometry'][:]
        except Exception:
            print(f'Error loading geometry file at {path}.')
            raise

        if preview_geometry:
            Geometry.preview(geometry.lung_tissue)
            #the pragram will be blocked even a new thread is created. maybe a conflict of the popping window?
            #thread = threading.Thread(target = Geometry.preview, args = (geometry.lung_tissue, ))
            #thread.start()
            
        return state

    # same as the super class. not needed for now
    # def advance(self, state: State, previous_time: float):
    #     """Advance the state by a single time step."""
    #     # do nothing since the geoemtry is constant
    #     return state

    @classmethod
    def preview(cls, grid: np.ndarray):
        #print(vtk.vtkVersion.GetVTKSourceVersion())
        #sys.stdout.flush()
        dataImporter = vtk.vtkImageImport()

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
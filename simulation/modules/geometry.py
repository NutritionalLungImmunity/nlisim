from enum import Enum

import attr
import h5py
import numpy as np
from scipy.sparse import csr_matrix
import vtk

from simulation.module import Module, ModuleState
from simulation.state import grid_variable, State
from simulation.validation import ValidationError


class TissueTypes(Enum):
    AIR = 0
    BLOOD = 1
    OTHER = 2
    EPITHELIUM = 3
    SURFACTANT = 4
    PORE = 5

    @classmethod
    def validate(cls, value: np.ndarray):
        return np.logical_and(value >= 0, value <= 5).all() and np.issubclass_(
            value.dtype.type, np.integer
        )


@attr.s(kw_only=True, repr=False)
class LaplacianMatrix(object):
    surf_lapl: csr_matrix = attr.ib(init=False)

    def save(self, group: h5py.Group, name: str, metadata: dict) -> h5py.Group:
        """Save an attribute into an HDF5 group."""
        surf_lapl = self.surf_lapl
        composite_group = group.create_group(name)
        composite_group.create_dataset('data', data=surf_lapl.data)
        composite_group.create_dataset('indptr', data=surf_lapl.indptr)
        composite_group.create_dataset('indices', data=surf_lapl.indices)
        composite_group.attrs['shape'] = surf_lapl.shape
        return composite_group

    @classmethod
    def create_matrix(cls):
        return LaplacianMatrix()


@attr.s(kw_only=True, repr=False)
class GeometryState(ModuleState):
    lung_tissue = grid_variable(np.dtype('int'))
    laplacian_matrix: LaplacianMatrix = attr.ib()

    @laplacian_matrix.default
    def __set_default_laplacian_matrix(self):
        return LaplacianMatrix.create_matrix()

    @lung_tissue.validator
    def _validate_lung_tissue(self, attribute: attr.Attribute, value: np.ndarray) -> None:
        if not TissueTypes.validate(value):
            raise ValidationError('input illegal')

    def __repr__(self):
        return 'GeometryState(lung_tissue)'


class Geometry(Module):
    name = 'geometry'
    defaults = {'geometry_path': 'geometry.hdf5', 'preview_geometry': 'False'}
    StateClass = GeometryState

    def initialize(self, state: State):
        geometry: GeometryState = state.geometry
        preview_geometry = self.config.getboolean('preview_geometry')
        path = self.config.get('geometry_path')
        try:
            with h5py.File(path, 'r') as f:
                if f['geometry'][:].shape != state.grid.shape:
                    raise ValidationError("shape doesn\'t match")
                geometry.lung_tissue[:] = f['geometry'][:]

                surf_lapl = f['surf_lapl']
                geometry.laplacian_matrix.surf_lapl = csr_matrix(
                    (surf_lapl['data'][:], surf_lapl['indices'][:], surf_lapl['indptr'][:]),
                    surf_lapl.attrs['shape'],
                )

        except Exception:
            print(f'Error loading geometry file at {path}.')
            raise

        if preview_geometry:
            Geometry.preview(geometry.lung_tissue)

        return state

    @classmethod
    def preview(cls, grid: np.ndarray):
        data_importer = vtk.vtkImageImport()

        xbin = grid.shape[2]
        ybin = grid.shape[1]
        zbin = grid.shape[0]
        g = np.uint8(grid.reshape(grid.shape[0] * grid.shape[1] * grid.shape[2]))

        data_string = g.tostring()
        data_importer.CopyImportVoidPointer(data_string, len(data_string))
        data_importer.SetDataScalarTypeToUnsignedChar()
        data_importer.SetNumberOfScalarComponents(1)

        data_importer.SetDataExtent(0, xbin - 1, 0, ybin - 1, 0, zbin - 1)
        data_importer.SetWholeExtent(0, xbin - 1, 0, ybin - 1, 0, zbin - 1)

        # Create transfer mapping scalar value to opacity
        opacity_transfer_function = vtk.vtkPiecewiseFunction()
        opacity_transfer_function.AddPoint(0, 0.0)
        opacity_transfer_function.AddPoint(1, 0.2)
        opacity_transfer_function.AddPoint(2, 0.005)
        opacity_transfer_function.AddPoint(3, 1)
        opacity_transfer_function.AddPoint(4, 0.2)
        opacity_transfer_function.AddPoint(5, 0.2)

        # Create transfer mapping scalar value to color
        color_transfer_function = vtk.vtkColorTransferFunction()
        color_transfer_function.AddRGBPoint(0, 0.0, 0.0, 1.0)
        color_transfer_function.AddRGBPoint(1, 1.0, 0.0, 0.0)
        color_transfer_function.AddRGBPoint(2, 0.0, 0.0, 1.0)
        color_transfer_function.AddRGBPoint(3, 1.0, 1.0, 1.0)
        color_transfer_function.AddRGBPoint(4, 1.0, 1.0, 1.0)
        color_transfer_function.AddRGBPoint(5, 1.0, 1.0, 1.0)

        # The property describes how the data will look
        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(color_transfer_function)
        volume_property.SetScalarOpacity(opacity_transfer_function)
        # volumeProperty.ShadeOn()
        volume_property.SetInterpolationTypeToLinear()

        # The mapper / ray cast function know how to render the data
        volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        volume_mapper.SetBlendModeToComposite()
        volume_mapper.SetInputConnection(data_importer.GetOutputPort())

        # The volume holds the mapper and the property and
        # can be used to position/orient the volume
        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)

        ren = vtk.vtkRenderer()
        ren_win = vtk.vtkRenderWindow()
        ren_win.AddRenderer(ren)
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(ren_win)

        ren.AddVolume(volume)
        ren.SetBackground(1, 1, 1)
        ren_win.SetSize(600, 600)
        ren_win.Render()

        def check_abort(obj, event):
            if obj.GetEventPending() != 0:
                obj.SetAbortRender(1)

        ren_win.AddObserver('AbortCheckEvent', check_abort)

        iren.Initialize()
        ren_win.Render()
        iren.Start()

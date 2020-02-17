import json
import struct
import time
from typing import List, Tuple, Union

import h5py
import numpy as np
from scipy import ndimage
import vtk

from simulation.coordinates import Point
from simulation.diffusion import discrete_laplacian
from simulation.geometry.math_function import Cylinder, Sphere
from simulation.grid import RectangularGrid

# tissue type
SAC = 'sac'
DUCT = 'duct'
QUADRIC = 'quadric'

# tissue number
AIR = 0
BLOOD = 1
REGULAR_TISSUE = 2
EPITHELIUM = 3
SURF = 4
PORES = 5

ShapeType = Tuple[int, int, int]
SpacingType = Tuple[float, float, float]


class Geometry(object):
    def __init__(self, shape: ShapeType, space: SpacingType):
        self.shape = shape
        self.space = space
        self.grid = RectangularGrid.construct_uniform(shape, space)
        self.geo = self.grid.allocate_variable(dtype=np.int8)
        self.geo.fill(2)
        self.fixed = np.zeros(shape)

        self.duct_f: List[Union[Sphere, Cylinder]] = []
        self.sac_f: List[Union[Sphere, Cylinder]] = []

    def scaling(self, n):
        for f in self.duct_f:
            f.scale(n)

        for f in self.sac_f:
            f.scale(n)

        self.xbin = self.xbin * n
        self.ybin = self.ybin * n
        self.zbin = self.zbin * n

        self.geo = np.zeros((self.zbin, self.ybin, self.xbin))
        self.geo.fill(2)
        self.fixed = np.zeros((self.zbin, self.ybin, self.xbin))

    def add(self, function):
        if function.type == SAC:
            self.sac_f.append(function)
        elif function.type == DUCT:
            self.duct_f.append(function)
        else:
            raise Exception('Unknown tissue type')

    def construct_sphere(self, lungtissue, center: Point, r: float):
        coords = np.ogrid[: lungtissue.shape[0], : lungtissue.shape[1], : lungtissue.shape[2]]
        distance = np.sqrt(
            (coords[0] - center.z) ** 2 + (coords[1] - center.y) ** 2 + (coords[2] - center.x) ** 2
        )
        return 1 * (distance <= r)

    def construct_cylinder(
        self, lungtissue, center: Point, length: float, direction: np.ndarray, r: float
    ):
        line = (direction < 1).astype(int)
        coords = np.ogrid[: lungtissue.shape[0], : lungtissue.shape[1], : lungtissue.shape[2]]
        distance = np.sqrt(
            line[0] * (coords[0] - center.z) ** 2
            + line[1] * (coords[1] - center.y) ** 2
            + line[2] * (coords[2] - center.x) ** 2
        )
        domin = direction * length
        mask = (distance <= r).astype(int)
        mask[domin[0] :, domin[1] :, domin[2] :] = 0
        return mask

    def construct(self):
        tissue = self.geo
        fixed = self.fixed
        print(tissue.shape)

        print('constructing air duct')
        # construct air duct
        for function in self.duct_f:
            if isinstance(function, Cylinder):

                air_mask = self.construct_cylinder(
                    tissue, function.center, function.length, function.direction, function.radius
                )

                epi_mask = self.construct_cylinder(
                    tissue,
                    function.center,
                    function.length,
                    function.direction,
                    function.radius + 1,
                )
                tissue[np.logical_and(epi_mask == 1, fixed == 0)] = EPITHELIUM
                tissue[np.logical_and(air_mask == 1, fixed == 0)] = AIR
                fixed[air_mask == 1] = 1

        print('constructing alveolus')
        # construct sac
        for function in self.sac_f:
            if isinstance(function, Sphere):
                air_mask = self.construct_sphere(tissue, function.center, function.radius)
                epi_mask = self.construct_sphere(tissue, function.center, function.radius + 1)
                tissue[np.logical_and(epi_mask == 1, fixed == 0)] = EPITHELIUM
                tissue[np.logical_and(air_mask == 1, fixed == 0)] = AIR
                fixed[epi_mask == 1] = 1

        print('constructing surfactant layer and capillary')
        # construct surfactant and blood vessel
        epi_mask = np.where(tissue == EPITHELIUM, 2, 0)
        surf_mask = ndimage.filters.convolve(epi_mask, np.ones((3, 3, 3)))
        tissue[np.logical_and(tissue == AIR, surf_mask > 0)] = SURF
        tissue[np.logical_and(tissue == REGULAR_TISSUE, surf_mask > 0)] = BLOOD

    def write_to_vtk(self, filename='geometry.vtk'):
        zbin, ybin, xbin = self.shape

        f = open(filename, 'w')
        f.write('# vtk DataFile Version 4.2\n')
        f.write('Aspergillus simulation: Geometry\n')
        f.write('BINARY\n')
        f.write('DATASET STRUCTURED_POINTS\n')
        f.write('DIMENSIONS ' + str(xbin) + ' ' + str(ybin) + ' ' + str(zbin) + '\n')
        f.write('ASPECT_RATIO 1 1 1\n')
        f.write('ORIGIN 0 0 0\n')
        f.write('POINT_DATA ' + str(xbin * ybin * zbin) + '\n')
        f.write('SCALARS TissueType unsigned_char 1\n')
        f.write('LOOKUP_TABLE default\n')
        f.close()

        f = open(filename, 'ab')
        array = self.geo.flatten()
        array = array.astype(int)

        b = struct.pack(len(array) * 'B', *array)
        f.write(b)
        f.close()

    def write_to_hdf5(self, filename):

        # surfactant layer laplacian
        surf_lapl = discrete_laplacian(self.grid, self.geo == SURF)

        # Write data to HDF5
        with h5py.File(filename, 'w') as data_file:
            data_file.create_dataset('geometry', data=self.geo)
            matrix = data_file.create_group('surf_lapl')
            matrix.create_dataset('data', data=surf_lapl.data)
            matrix.create_dataset('indptr', data=surf_lapl.indptr)
            matrix.create_dataset('indices', data=surf_lapl.indices)
            matrix.attrs['shape'] = surf_lapl.shape

    def preview(self):
        zbin, ybin, xbin = self.shape
        data_importer = vtk.vtkImageImport()

        g = self.geo.flatten()
        g = np.uint8(g)
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

        # Create transfer mapping scalar value to color
        color_transfer_function = vtk.vtkColorTransferFunction()
        color_transfer_function.AddRGBPoint(0, 0.0, 0.0, 1.0)
        color_transfer_function.AddRGBPoint(1, 1.0, 0.0, 0.0)
        color_transfer_function.AddRGBPoint(2, 0.0, 0.0, 1.0)
        color_transfer_function.AddRGBPoint(3, 1.0, 1.0, 1.0)

        # The property describes how the data will look
        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(color_transfer_function)
        volume_property.SetScalarOpacity(opacity_transfer_function)
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


def generate_geometry(config, output, preview):
    start_time = time.time()

    with open(config) as f:
        data = json.load(f)

        shape = (data['shape']['zbin'], data['shape']['ybin'], data['shape']['xbin'])

        space = (data['space']['dz'], data['space']['dy'], data['space']['dx'])

        g = Geometry(shape, space)

        for function in data['function']:

            if function['shape'] == 'sphere':
                f = Sphere(function['center'], function['radius'], function['type'])
                g.add(f)

            elif function['shape'] == 'cylinder':
                f = Cylinder(
                    function['center'],
                    function['direction'],
                    function['radius'],
                    function['length'],
                    function['type'],
                )
                g.add(f)

        # g.scaling(data["scaling"])
        g.construct()
        g.write_to_hdf5(output)
    print(f'--- {(time.time() - start_time)} seconds ---')
    if preview:
        g.preview()

import json
import struct
import time
from typing import List, Tuple, Union

import h5py
import numpy as np
from scipy import ndimage
import vtk

from nlisim.coordinates import Point
from nlisim.diffusion import discrete_laplacian
from nlisim.geometry.math_function import Cylinder, Sphere
from nlisim.grid import RectangularGrid

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
    def __init__(self, shape: ShapeType, space: SpacingType, scale: int, randomness: int):
        self.scale = scale
        self.randomness = randomness
        self.shape = (shape[0] * scale, shape[1] * scale, shape[2] * scale)
        self.space = space
        self.grid = RectangularGrid.construct_uniform(self.shape, self.space)

        self.geo = self.grid.allocate_variable(dtype=np.dtype(np.int8))
        self.geo.fill(2)
        self.fixed = np.zeros(self.shape)

        self.duct_f: List[Union[Sphere, Cylinder]] = []
        self.sac_f: List[Union[Sphere, Cylinder]] = []

    def add(self, function):
        """Add functions to the generator."""
        function.scale(self.scale)
        if function.type == SAC:
            self.sac_f.append(function)
        elif function.type == DUCT:
            self.duct_f.append(function)
        else:
            raise Exception('Unknown tissue type')

    def construct_sphere(self, lungtissue, center: Point, r: float):
        """Construct sphere within simulation space."""
        coords = np.ogrid[: lungtissue.shape[0], : lungtissue.shape[1], : lungtissue.shape[2]]
        distance = np.sqrt(
            (coords[0] - center.z) ** 2 + (coords[1] - center.y) ** 2 + (coords[2] - center.x) ** 2
        )
        return 1 * (distance <= r)

    def construct_cylinder(
        self,
        lung_tissue: np.ndarray,
        center: Point,
        length: float,
        direction: np.ndarray,
        r: np.ndarray,
    ):
        """Construct cylinder within simulation space."""
        coords = np.indices(lung_tissue.shape, dtype=np.float64).T

        # normalize direction, just in case
        direction = direction / np.linalg.norm(direction)

        relative_coords: np.ndarray = coords - center
        distance_along_axis: np.ndarray = relative_coords @ direction
        distance_from_axis = np.linalg.norm(
            relative_coords - np.multiply.outer(relative_coords @ direction, direction), axis=3
        )
        mask = np.logical_and(distance_from_axis <= r.T, distance_along_axis <= (length / 2.0)).T
        return mask

    def construct_air_duct(self, random_mask):
        print('constructing air duct...')
        tissue = self.geo
        fixed = self.fixed
        # construct air duct
        for function in self.duct_f:
            if isinstance(function, Cylinder):
                air_mask = self.construct_cylinder(
                    tissue,
                    function.center,
                    function.length,
                    function.direction,
                    function.radius + random_mask,
                )
                tissue[np.logical_and(air_mask == 1, fixed == 0)] = AIR

        # blur the noise to maintain the continuousness of the air
        blur_mask = np.where(tissue == AIR, 1, 0)
        blur_air_mask = ndimage.filters.convolve(blur_mask, np.ones((3, 3, 3)))
        tissue[blur_air_mask > 13] = AIR
        tissue[blur_air_mask <= 13] = REGULAR_TISSUE
        fixed[tissue == AIR] = 1

        # construct epithelium layer
        air_mask = np.where(tissue == AIR, 1, 0)
        epithelium_mask = ndimage.filters.convolve(air_mask, np.ones((3, 3, 3)))
        tissue[np.logical_and(epithelium_mask > 0, tissue == REGULAR_TISSUE)] = EPITHELIUM

    def construct_alveolus(self, random_mask):
        tissue = self.geo
        fixed = self.fixed
        print('constructing alveolus...')
        # construct sac
        for function in self.sac_f:
            if isinstance(function, Sphere):
                air_mask = self.construct_sphere(
                    tissue, function.center, function.radius + random_mask
                )
                blur_air_mask = ndimage.filters.convolve(air_mask, np.ones((3, 3, 3)))
                fixed_air_mask = np.logical_and(blur_air_mask > 13, fixed == 0)
                tissue[fixed_air_mask] = AIR
                tissue[
                    np.logical_and(np.logical_and(blur_air_mask <= 13, fixed == 0), tissue == AIR)
                ] = REGULAR_TISSUE
                fixed[fixed_air_mask] = 1

                # construct epithelium layer
                epithelium_mask = ndimage.filters.convolve(fixed_air_mask, np.ones((3, 3, 3)))
                fixed_epithelium_mask = np.logical_and(epithelium_mask > 0, fixed == 0)
                tissue[fixed_epithelium_mask] = EPITHELIUM
                fixed[fixed_epithelium_mask] = 1

    def construct(self, simple):
        """Construct the simulation space with math functions."""
        tissue = self.geo
        # fixed = self.fixed

        random_mask = np.random.normal(0, self.randomness, self.shape)

        self.construct_air_duct(random_mask)
        self.construct_alveolus(random_mask)

        epi_mask = np.where(tissue == EPITHELIUM, 2, 0)
        surf_mask = ndimage.filters.convolve(epi_mask, np.ones((3, 3, 3)))
        print('constructing surfactant layer and capillary...')
        # construct surfactant and blood vessel
        if not simple:
            tissue[np.logical_and(tissue == AIR, surf_mask > 0)] = SURF
        tissue[np.logical_and(tissue == REGULAR_TISSUE, surf_mask > 0)] = BLOOD

    def write_to_vtk(self, filename):
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

    def write_to_hdf5(self, filename, laplacian):
        # Write data to HDF5
        with h5py.File(filename, 'w') as data_file:
            data_file.create_dataset('geometry', data=self.geo)

            if laplacian:
                # embed laplacian matrix for all layers
                # surfactant layer laplacian
                surf_lapl = discrete_laplacian(self.grid, self.geo == SURF)
                # epithelium layer laplacian
                epi_lapl = discrete_laplacian(self.grid, self.geo == EPITHELIUM)
                # capillary layer laplacian
                blood_lapl = discrete_laplacian(self.grid, self.geo == BLOOD)

                d = {'surf_lapl': surf_lapl, 'epi_lapl': epi_lapl, 'blood_lapl': blood_lapl}
                matrices = data_file.create_group('lapl_matrices')
                for name, lapl in d.items():
                    matrix = matrices.create_group(name)
                    matrix.create_dataset('data', data=lapl.data)
                    matrix.create_dataset('indptr', data=lapl.indptr)
                    matrix.create_dataset('indices', data=lapl.indices)
                    matrix.attrs['shape'] = lapl.shape

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


def generate_geometry(config, output, preview, simple, lapl):
    start_time = time.time()

    with open(config) as f:
        data = json.load(f)

        scale = data['scaling']
        randomness = data['randomness']
        shape = (data['shape']['zbin'], data['shape']['ybin'], data['shape']['xbin'])
        space = (data['space']['dz'], data['space']['dy'], data['space']['dx'])

        g = Geometry(shape, space, scale, randomness)

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
        g.construct(simple)
        g.write_to_hdf5(output + '.hdf5', lapl)
        g.write_to_vtk(output + '.vtk')
    print(f'--- {(time.time() - start_time)} seconds ---')
    if preview:
        g.preview()

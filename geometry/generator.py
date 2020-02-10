import numpy as np
import json
import random
import struct
import sys
from multiprocessing import Array, Condition, Manager, Process
import math
import time
import ctypes
import os
import random
import optparse
from vtk import *
from vtk.util import numpy_support
import h5py

# tissue type
SAC = "sac"
DUCT = "duct"
QUADRIC = "quadric"

# tissue number
AIR = 0
BLOOD = 1
REGULAR_TISSUE = 2
EPITHELIUM = 3
MMLAYER = 4 
PORES = 5 # macrophage movement layer

BLOOD_VESSEL_LAYER = 1

# construct code
CONSTRUCT_DUCT = 0
CONSTRUCT_SAC = 1
CONSTRUCT_AIR = 0
CONSTRUCT_EPI = 1
CONSTRUCT_VESSEL = 2

# process state
PROCESSING = 0
READY = 1

SQRT_ROOT_2 = math.sqrt(2)

class Quadric():
    def __init__(self, json):
        self.set_coef(json["cx"], json["cy"], json["cz"], json["r"])
        self.set_shift(json["a"], json["b"], json["c"])
        self.set_range(json["x_min"], json["x_max"], json["y_min"], json["y_max"], json["z_min"], json["z_max"])
        self.tissue_type = json["tissue_type"]

    def set_coef(self, cx, cy, cz, r):
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.r = r

    def set_shift(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def set_range(self, x_min, x_max, y_min, y_max, z_min, z_max):
        self.x_max = x_max
        self.x_min = x_min
        self.y_max = y_max
        self.y_min = y_min
        self.z_min = z_min
        self.z_max = z_max

    def scaling(self, n):
        self.a = self.a * n
        self.b = self.b * n
        self.c = self.c * n
        self.r = self.r * n

        self.x_max = self.x_max * n
        self.x_min = self.x_min * n
        self.x_max = self.x_max * n
        self.y_max = self.y_max * n
        self.y_min = self.y_min * n
        self.z_min = self.z_min * n
        self.z_max = self.z_max * n

class Vector():
    def __init__(self, json):
        self.xt = np.poly1d(json["xt"])
        self.yt = np.poly1d(json["yt"])
        self.zt = np.poly1d(json["zt"])
        self.r = json["r"]
        self.t_min = json["t_min"]
        self.t_max = json["t_max"]
        self.set_range(json["x_min"], json["x_max"], json["y_min"], json["y_max"], json["z_min"], json["z_max"])
        self.tissue_type = json["tissue_type"]

    def set_range(self, x_min, x_max, y_min, y_max, z_min, z_max):
        self.x_max = x_max
        self.x_min = x_min
        self.y_max = y_max
        self.y_min = y_min
        self.z_min = z_min
        self.z_max = z_max

    def get_val(self, t):
        return [self.xt(t), self.yt(t), self.zt(t)]

    def scaling(self, n):
        self.xt = self.xt * n
        self.yt = self.yt * n
        self.zt = self.zt * n
        self.r = self.r * n

        self.x_max = self.x_max * n
        self.x_min = self.x_min * n
        self.x_max = self.x_max * n
        self.y_max = self.y_max * n
        self.y_min = self.y_min * n
        self.z_min = self.z_min * n
        self.z_max = self.z_max * n


class Geometry():
    def __init__(self, xbin = 0, ybin = 0, zbin = 0, multi_process = 0, vessel_layer_json = None):
        self.xbin = xbin
        self.ybin = ybin
        self.zbin = zbin

        self.host = ''
        self.redis_port = ''
        self.redis_pwd = ''

        self.geo = Array(ctypes.c_double, xbin * ybin * zbin)
        self.overwrite = Array(ctypes.c_bool, xbin * ybin * zbin)
        self.lock = Array(ctypes.c_double, multi_process)

        if vessel_layer_json != None:
            self.set_vessel_layer_params(vessel_layer_json)


        self.multi_process = multi_process
        g = np.frombuffer(self.geo.get_obj())
        g.fill(2)

        o = np.frombuffer(self.overwrite.get_obj(), dtype=bool)
        o.fill(True)

        for i in range(len(self.lock)):
            self.lock[i] = PROCESSING

        self.duct_f = [] # function list
        self.sac_f = []

    def set_http_redis(self, redis_url, redis_port, redis_password):
        self.host = redis_url
        self.redis_port = redis_port
        self.redis_pwd = redis_password

    def set_vessel_layer_params(self, json):
        self.vessel_xmin = json["x_min"]
        self.vessel_xmax = json["x_max"]

        self.vessel_ymin = json["y_min"]
        self.vessel_ymax = json["y_max"]

        self.vessel_zmin = json["z_min"]
        self.vessel_zmax = json["z_max"]

        self.interstitium = json["interstitium"]

    def scaling(self, n):
        for f in self.duct_f:
            f.scaling(n)

        for f in self.sac_f:
            f.scaling(n)

        self.xbin = self.xbin * n
        self.ybin = self.ybin * n
        self.zbin = self.zbin * n

        self.geo = Array(ctypes.c_double, self.xbin * self.ybin * self.zbin)
        g = np.frombuffer(self.geo.get_obj())
        g.fill(REGULAR_TISSUE)

        self.overwrite = Array(ctypes.c_bool, self.xbin * self.ybin * self.zbin)
        o = np.frombuffer(self.overwrite.get_obj(), dtype=bool)
        o.fill(True)

        self.vessel_xmin = self.vessel_xmin * n
        self.vessel_xmax = self.vessel_xmax * n

        self.vessel_ymin = self.vessel_ymin * n
        self.vessel_ymax = self.vessel_ymax * n

        self.vessel_zmin = self.vessel_zmin * n
        self.vessel_zmax = self.vessel_zmax * n


    # check if a point is in function's domain
    def in_range(self, x, y ,z, function):
        return x >= function.x_min and x < function.x_max and y >= function.y_min and y < function.y_max and z >= function.z_min and z < function.z_max

    # check if a point is inside the geometry space
    def in_range_geo(self, x, y, z):
        return x >= 0 and x < self.xbin and y >= 0 and y < self.ybin and z >= 0 and z < self.zbin

    # return the distance between two points
    def distance(self, x1, x2, y1, y2, z1, z2):
        return sqrt( (x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

    def add(self, function):
        if function.tissue_type == SAC:
            self.sac_f.append(function)
        elif function.tissue_type == DUCT:
            self.duct_f.append(function)
        else:
            raise Exception("Unknown tissue type")

    def construct(self):
        section = math.floor(self.xbin / self.multi_process)
        current = 0
        processes = []

        manager = Manager()
        condition = manager.Condition()

        for i in range(self.multi_process):

            

            if i == self.multi_process - 1: # last section
                #arg = [current, self.xbin, 0, self.ybin, 0, self.zbin, i, condition]
                processes.append(Process(target = self.construct_multi, args = (current, self.xbin, 0, self.ybin, 0, self.zbin, i, condition,)))

            else:
                #arg = [current, current + section, 0, self.ybin, 0, self.zbin, i, condition]
                processes.append(Process(target = self.construct_multi, args = (current, current + section, 0, self.ybin, 0, self.zbin, i, condition,)))
            #args.append(arg)
            current += section

        #ProcessingPool().map(self.construct_multi, args)

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        self.macrophage_movement_layer()


    def test_and_wait(self, condition):
        lock = np.frombuffer(self.lock.get_obj())

        wait = False

        with condition:
            for i in lock:
                if i != READY:
                    wait = True
                    # print(str(os.getpid()) + " waiting")
                    # sys.stdout.flush()
                    condition.wait()
                    break

            if wait == False:
                for i in range(len(lock)):
                    lock[i] = PROCESSING

                #print(str(os.getpid()) + " notifying")
                condition.notify_all()


    def construct_multi(self, x_min, x_max, y_min, y_max, z_min, z_max, id, condition):
        start_time = time.time()
        
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                for z in range(z_min, z_max):
                    for function in self.duct_f:
                        self.check_geometry_type(function, x, y, z, CONSTRUCT_DUCT)

        lock = np.frombuffer(self.lock.get_obj())
        lock[id] += 1
        self.test_and_wait(condition)
        
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                for z in range(z_min, z_max):
                    for function in self.sac_f:
                        self.check_geometry_type(function, x, y, z, CONSTRUCT_SAC)

        lock = np.frombuffer(self.lock.get_obj())
        lock[id] += 1
        self.test_and_wait(condition)

        for function in self.duct_f:
            function.r += self.interstitium + 2
        for function in self.sac_f:
            function.r += self.interstitium + 2

        section = math.ceil((self.vessel_xmin + self.vessel_xmax) / self.multi_process)
        vessel_xmin = self.vessel_xmin + section * id
        vessel_xmax = vessel_xmin + section

        for x in range(vessel_xmin, vessel_xmax):
            for y in range(self.vessel_ymin, self.vessel_ymax):
                for z in range(self.vessel_zmin, self.vessel_zmax):
                    for function in self.duct_f:
                        self.check_geometry_type(function, x, y, z, CONSTRUCT_VESSEL)
                    for function in self.sac_f:
                        self.check_geometry_type(function, x, y, z, CONSTRUCT_VESSEL)

        #print("--- process: " + str(os.getpid()) + " ends in %s seconds ---" % (time.time() - start_time))

    def check_geometry_type(self, function, x, y, z, code):
        #print(type(function))
        if self.in_range(x, y, z, function):
            if (type(function) is Quadric):

                d = function.cx * (x + function.a) ** 2 + function.cy * (y + function.b) ** 2 + function.cz * (z + function.c) ** 2

                if code == CONSTRUCT_VESSEL:
                    if d <= function.r ** 2 and d > (function.r - 1) ** 2:
                        self.change_tissue_type(function, x, y, z, code, None)

                else:
                    if d <= function.r ** 2:
                        self.change_tissue_type(function, x, y, z, code, CONSTRUCT_AIR)
                    elif d <= (function.r + 1) ** 2:
                        self.change_tissue_type(function, x, y, z, code, CONSTRUCT_EPI)

            elif (type(function) is Vector):
                xt = function.xt - np.poly1d([x])
                xt = xt * xt

                yt = function.yt - np.poly1d([y])
                yt = yt * yt

                zt = function.zt - np.poly1d([z])
                zt = zt * zt

                p = xt + yt + zt
                p = p.deriv()

                #root = queue.Queue()
                root = p.r[:]
                np.append(root, function.t_min)
                np.append(root, function.t_max)

                for r in root:
                    #print(r)
                    if r >= function.t_min and r <= function.t_max and self.distance(x, function.xt(r), y, function.yt(r), z, function.zt(r)) <= function.r:
                        self.change_tissue_type(function, x, y, z, code, tissue)
                        break

    def change_tissue_type(self, function, x, y, z, code, tissue):
        g = np.frombuffer(self.geo.get_obj())
        g = g.reshape(self.zbin,self.ybin,self.xbin).transpose()

        overwrite = np.frombuffer(self.overwrite.get_obj(), dtype = bool)
        overwrite = overwrite.reshape(self.zbin, self.ybin, self.xbin).transpose()

        if code == CONSTRUCT_DUCT:
            if tissue == CONSTRUCT_AIR:
                g[x][y][z] = AIR
                overwrite[x][y][z] = False

            elif tissue == CONSTRUCT_EPI and overwrite[x][y][z]:
                g[x][y][z] = EPITHELIUM

            else:
                raise Exception("unknown tissue type")

        elif code == CONSTRUCT_SAC:
            if tissue == CONSTRUCT_AIR and overwrite[x][y][z]:
                g[x][y][z] = AIR
                overwrite[x][y][z] = False
                #print (x,y,z)
            elif tissue == CONSTRUCT_EPI and overwrite[x][y][z]:
                g[x][y][z] = EPITHELIUM
                overwrite[x][y][z] = False

        elif code == CONSTRUCT_VESSEL:
            if g[x][y][z] == REGULAR_TISSUE:
                g[x][y][z] = BLOOD

    def macrophage_movement_layer(self):
        g = np.frombuffer(self.geo.get_obj())
        g = g.reshape(self.zbin,self.ybin,self.xbin).transpose()
        pores_cnt = 0
        mmlayer_cnt = 0

        for x in range(self.xbin):
            for y in range(self.ybin):
                for z in range(self.zbin): 
                    if g[x][y][z] == EPITHELIUM:
                        if (((x - 1 >= 0 and x + 1 < self.xbin and g[x-1][y][z] == g[x+1][y][z] == AIR)
                            or (y - 1 >= 0 and y + 1 < self.ybin and g[x][y-1][z] == g[x][y+1][z] == AIR)
                            or (z - 1 >= 0 and z + 1 < self.zbin and g[x][y][z-1] == g[x][y][z+1] == AIR)) 
                            and random.random() < 0.05 and not self.near_tissue(x,y,z,g,REGULAR_TISSUE)):
                            g[x][y][z] = PORES
                            pores_cnt += 1

        for x in range(self.xbin):
            for y in range(self.ybin):
                for z in range(self.zbin):
                    #isMacroLayer = False
                    if (g[x][y][z] == AIR): #is a air space
                        #check the voxels nearby
                        if self.near_tissue(x,y,z,g,EPITHELIUM):
                            g[x][y][z] = MMLAYER
                            mmlayer_cnt += 1

    def near_tissue(self, x, y, z, geo, tissue_type):
        for i in range(x-1, x+2):
            for j in range(y-1, y+2):
                for k in range(z-1, z+2):
                    if ((i!=x or j!=y or k!=z) 
                        and (i >= 0 and i< self.xbin)
                        and (j >= 0 and j< self.ybin)
                        and (k >= 0 and k< self.zbin)): #in range
                            if geo[i][j][k] == tissue_type:
                                return True
        return False

    def write_to_vtk(self, filename = "geometry.vtk"):
        f = open(filename, "w")
        f.write("# vtk DataFile Version 4.2\n")
        f.write("Aspergillus simulation: Geometry\n")
        f.write("BINARY\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write("DIMENSIONS " + str(self.xbin) + " " + str(self.ybin) + " " + str(self.zbin) + "\n")
        f.write("ASPECT_RATIO 1 1 1\n")
        f.write("ORIGIN 0 0 0\n")
        f.write("POINT_DATA " + str(self.xbin * self.ybin * self.zbin) + "\n")
        f.write("SCALARS TissueType unsigned_char 1\n")
        f.write("LOOKUP_TABLE default\n")
        f.close()


        f = open(filename, "ab")
        array = np.frombuffer(self.geo.get_obj())
        array = array.astype(int)

        b = struct.pack(len(array) * 'B', *array)
        f.write(b)
        f.close()

    def write_to_npy(self, filename):
        print("writing to npy ...")
        sys.stdout.flush()
        lungtissue = np.frombuffer(self.geo.get_obj())
        lungtissue = lungtissue.reshape(self.zbin, self.ybin, self.xbin)
        np.save(filename, lungtissue)

    def write_to_hdf5(self, filename):
        import h5py
        lungtissue = np.frombuffer(self.geo.get_obj())
        g = lungtissue.reshape(self.zbin, self.ybin, self.xbin)

        # Write data to HDF5
        with h5py.File(filename, 'w') as data_file:
            data_file.create_dataset('geometry', data=g)

    def load_from_vtk(self, filename):

        print("Loading VTK file %s ..." % (filename))
        sys.stdout.flush()

        reader = vtkStructuredPointsReader()
        reader.SetFileName(filename)
        reader.ReadAllVectorsOn()
        reader.ReadAllScalarsOn()
        reader.Update()

        data = reader.GetOutput()
        #print(data.GetPointData())
        points = numpy_support.vtk_to_numpy(data.GetPointData().GetArray('TissueType'))
        
        self.geo = Array(ctypes.c_double, data.GetNumberOfPoints())
        g = np.frombuffer(self.geo.get_obj())



        for i in range(data.GetNumberOfPoints()):
            g[i] = points[i]

        dim = data.GetDimensions()
        self.xbin = dim[0]
        self.ybin = dim[1]
        self.zbin = dim[2]

        print("Dimension: " + str(dim))
        print("Number of Points: " + str(data.GetNumberOfPoints()))

        sys.stdout.flush()

    def load_from_hdf5(self, filename):
        with h5py.File('geometry.hdf5', 'r') as f:
            g = f['geometry'][:]
            self.xbin = g.shape[2]
            self.ybin = g.shape[1]
            self.zbin = g.shape[0]
            g = f['geometry'][:].reshape(self.zbin * self.ybin * self.zbin)
            self.geo = Array(ctypes.c_double, self.xbin * self.ybin * self.zbin)
            geo = np.frombuffer(self.geo.get_obj())

            for i in range(self.zbin * self.ybin * self.zbin):
                geo[i] = g[i] 

    def preview(self):
        print(vtk.vtkVersion.GetVTKSourceVersion())

        dataImporter = vtk.vtkImageImport()

        g = np.frombuffer(self.geo.get_obj())
        g = np.uint8(g)
        data_string = g.tostring()
        dataImporter.CopyImportVoidPointer(data_string, len(data_string))
        dataImporter.SetDataScalarTypeToUnsignedChar()
        dataImporter.SetNumberOfScalarComponents(1)

        dataImporter.SetDataExtent(0, self.xbin-1, 0, self.ybin-1, 0, self.zbin-1)
        dataImporter.SetWholeExtent(0, self.xbin-1, 0, self.ybin-1, 0, self.zbin-1)

        # Create transfer mapping scalar value to opacity
        opacityTransferFunction = vtk.vtkPiecewiseFunction()
        opacityTransferFunction.AddPoint(0, 0.0)
        opacityTransferFunction.AddPoint(1, 0.2)
        opacityTransferFunction.AddPoint(2, 0.005)
        opacityTransferFunction.AddPoint(3, 1)


        # Create transfer mapping scalar value to color
        colorTransferFunction = vtk.vtkColorTransferFunction()
        colorTransferFunction.AddRGBPoint(0, 0.0, 0.0, 1.0)
        colorTransferFunction.AddRGBPoint(1, 1.0, 0.0, 0.0)
        colorTransferFunction.AddRGBPoint(2, 0.0, 0.0, 1.0)
        colorTransferFunction.AddRGBPoint(3, 1.0, 1.0, 1.0)

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

def select_output(geometry, options):

    if options.write_to_vtk:
        if options.output_filename != None:
            geometry.write_to_vtk(options.output_filename)
        else:
            print("require output file name, see --help")


    if options.write_to_npy:
        if options.output_filename != None:
            geometry.write_to_npy(options.output_filename)
        else:
            print("require output file name, see --help")

    if options.write_to_hdf5:
        if options.output_filename != None:
            geometry.write_to_hdf5(options.output_filename)
        else:
            print("require output file name, see --help")

def initialize_opt():
    parser = optparse.OptionParser()

    parser.add_option('-i', '--input', dest = 'input_filename')
    parser.add_option('-o', '--output', dest = 'output_filename')

    parser.add_option('-u', '--url', dest = 'redis_url')
    parser.add_option('-p', '--port', dest = 'redis_port')
    parser.add_option('--pwd', dest = 'redis_pwd')

    parser.add_option('--preview', action = 'store_true', default = False, dest = 'preview')

    parser.add_option('--rjson', action = 'store_true', default = False, dest = "read_from_json")
    parser.add_option('--rvtk', action = 'store_true', default = False, dest = "read_from_vtk")
    parser.add_option('--rredis', action = 'store_true', default = False, dest = "read_from_redis")
    parser.add_option('--rhdf5', action = 'store_true', default=False, dest = "read_from_hdf5")

    parser.add_option('--wvtk', action = 'store_true', default = False, dest = 'write_to_vtk')
    parser.add_option('--wredis', action = 'store_true', default = False, dest = "write_to_redis")
    parser.add_option('--wnpy', action = 'store_true', default = False, dest = 'write_to_npy')
    parser.add_option('--whdf5', action = 'store_true', default = False, dest = 'write_to_hdf5')

    return parser

def generate_geometry(config):
    start_time = time.time()

    g = Geometry()

    with open(config) as f:
        data = json.load(f)

        dimen = data["dimension"]
        g = Geometry(dimen["xbin"], dimen["ybin"], dimen["zbin"], data["multi_process"], data["vessel layer"])

        #pprint(data)
        for function in data["function"]:
            
            if (function["type"] == QUADRIC):
                f = Quadric(function)
                g.add(f)
            elif (function["type"] == VECTOR):
                f = Vector(function)
                g.add(f)
        
        g.scaling(data["scaling"])
        g.construct()
        g.write_to_vtk('geometry.vtk')
        g.preview()
    print("--- %s seconds ---" % (time.time() - start_time))
        # select_output(g, options)
        #g.add_epithelium()
        #g.write_to_file(data["target"])
    '''
    elif (options.read_from_vtk):
        if (options.input_filename != None):
                g.load_from_vtk(options.input_filename)
                select_output(g, options)
        else:
            print("require input file name, see --help")


    elif (options.read_from_redis):
        g.load_from_redis(options.redis_url, options.redis_port, options.redis_pwd)
        select_output(g, options)

    elif (options.read_from_hdf5):
        if (options.input_filename != None):
                g.load_from_hdf5(options.input_filename)
                select_output(g, options)
        else:
            print("require input file name, see --help")

    else:
        print("require input source, see --help")
        return


    

    if options.preview:
        g.preview()
        '''

if __name__ == "__main__":
    main(sys.argv)

"""Created by Prabhu Ramachandran in Feb. 2008."""

import numpy
import vtk

# Useful constants for VTK arrays.
VTK_ID_TYPE_SIZE = vtk.vtkIdTypeArray().GetDataTypeSize()
if VTK_ID_TYPE_SIZE == 4:
    ID_TYPE_CODE = numpy.int32
elif VTK_ID_TYPE_SIZE == 8:
    ID_TYPE_CODE = numpy.int64

VTK_LONG_TYPE_SIZE = vtk.vtkLongArray().GetDataTypeSize()
if VTK_LONG_TYPE_SIZE == 4:
    LONG_TYPE_CODE = numpy.int32
    ULONG_TYPE_CODE = numpy.uint32
elif VTK_LONG_TYPE_SIZE == 8:
    LONG_TYPE_CODE = numpy.int64
    ULONG_TYPE_CODE = numpy.uint64


def get_vtk_array_type(numpy_array_type):
    """Return a VTK typecode given a numpy array."""
    # This is a Mapping from numpy array types to VTK array types.
    _np_vtk = {
        numpy.character: vtk.VTK_UNSIGNED_CHAR,
        numpy.uint8: vtk.VTK_UNSIGNED_CHAR,
        numpy.uint16: vtk.VTK_UNSIGNED_SHORT,
        numpy.uint32: vtk.VTK_UNSIGNED_INT,
        numpy.uint64: vtk.VTK_UNSIGNED_LONG_LONG,
        numpy.int8: vtk.VTK_CHAR,
        numpy.int16: vtk.VTK_SHORT,
        numpy.int32: vtk.VTK_INT,
        numpy.int64: vtk.VTK_LONG_LONG,
        numpy.float32: vtk.VTK_FLOAT,
        numpy.float64: vtk.VTK_DOUBLE,
        numpy.complex64: vtk.VTK_FLOAT,
        numpy.complex128: vtk.VTK_DOUBLE,
    }
    for key, vtk_type in _np_vtk.items():
        if (
            numpy_array_type == key
            or numpy.issubdtype(numpy_array_type, key)
            or numpy_array_type == numpy.dtype(key)
        ):
            return vtk_type
    raise TypeError('Could not find a suitable VTK type for %s' % (str(numpy_array_type)))


def get_vtk_to_numpy_typemap():
    """Return the VTK array type to numpy array type mapping."""
    _vtk_np = {
        vtk.VTK_BIT: numpy.bool,
        vtk.VTK_CHAR: numpy.int8,
        vtk.VTK_SIGNED_CHAR: numpy.int8,
        vtk.VTK_UNSIGNED_CHAR: numpy.uint8,
        vtk.VTK_SHORT: numpy.int16,
        vtk.VTK_UNSIGNED_SHORT: numpy.uint16,
        vtk.VTK_INT: numpy.int32,
        vtk.VTK_UNSIGNED_INT: numpy.uint32,
        vtk.VTK_LONG: LONG_TYPE_CODE,
        vtk.VTK_LONG_LONG: numpy.int64,
        vtk.VTK_UNSIGNED_LONG: ULONG_TYPE_CODE,
        vtk.VTK_UNSIGNED_LONG_LONG: numpy.uint64,
        vtk.VTK_ID_TYPE: ID_TYPE_CODE,
        vtk.VTK_FLOAT: numpy.float32,
        vtk.VTK_DOUBLE: numpy.float64,
    }
    return _vtk_np


def get_numpy_array_type(vtk_array_type):
    """Return a numpy array typecode given a VTK array type."""
    return get_vtk_to_numpy_typemap()[vtk_array_type]


def create_vtk_array(vtk_arr_type):
    """Create a VTK data array from another VTK array given the VTK array type."""
    return vtk.vtkDataArray.CreateDataArray(vtk_arr_type)


def numpy_to_vtk(num_array, deep=0, array_type=None):
    """Convert a real numpy Array to a VTK array object."""
    z = numpy.asarray(num_array)
    if not z.flags.contiguous:
        z = numpy.ascontiguousarray(z)

    shape = z.shape
    assert z.flags.contiguous, 'Only contiguous arrays are supported.'
    assert len(shape) < 3, 'Only arrays of dimensionality 2 or lower are allowed!'
    assert not numpy.issubdtype(z.dtype, numpy.dtype(complex).type), (
        'Complex numpy arrays cannot be converted to vtk arrays.'
        'Use real() or imag() to get a component of the array before'
        ' passing it to vtk.'
    )

    # First create an array of the right type by using the typecode.
    if array_type:
        vtk_typecode = array_type
    else:
        vtk_typecode = get_vtk_array_type(z.dtype)
    result_array = create_vtk_array(vtk_typecode)

    # Fixup shape in case its empty or scalar.
    try:
        assert shape[0] > 0
    except Exception:
        shape = (0,)

    # Find the shape and set number of components.
    if len(shape) == 1:
        result_array.SetNumberOfComponents(1)
    else:
        result_array.SetNumberOfComponents(shape[1])

    result_array.SetNumberOfTuples(shape[0])

    # Ravel the array appropriately.
    arr_dtype = get_numpy_array_type(vtk_typecode)
    if numpy.issubdtype(z.dtype, arr_dtype) or z.dtype == numpy.dtype(arr_dtype):
        z_flat = numpy.ravel(z)
    else:
        z_flat = numpy.ravel(z).astype(arr_dtype)
        # z_flat is now a standalone object with no references from the caller.
        # As such, it will drop out of this scope and cause memory issues if we
        # do not deep copy its data.
        deep = 1

    # Point the VTK array to the numpy data.  The last argument (1)
    # tells the array not to deallocate.
    result_array.SetVoidArray(z_flat, len(z_flat), 1)
    if deep:
        copy = result_array.NewInstance()
        copy.DeepCopy(result_array)
        result_array = copy
    else:
        result_array._numpy_reference = z
    return result_array

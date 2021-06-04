#!/usr/bin/env python

from __future__ import print_function
import sys
import os
import ctypes
import datetime

try:
   import numpy
except ImportError:
   print('Unable to import NumPy. Please ensure it is installed.')
   sys.exit(1)

# Determine potential names of dynamic library.
if os.name=='nt':
   dllpaths = ('otps2_.dll','libotps2_.dll')
elif os.name == "posix" and sys.platform == "darwin":
   dllpaths = ('libotps2_.dylib',)
else:
   dllpaths = ('libotps2_.so',)

def find_library(basedir):
    for dllpath in dllpaths:
        dllpath = os.path.join(basedir, dllpath)
        if os.path.isfile(dllpath):
            return dllpath

dllpath = find_library(os.path.dirname(os.path.abspath(__file__)))
if not dllpath:
    for basedir in sys.path:
        dllpath = find_library(basedir)
        if dllpath:
            break

if not dllpath:
   print('Unable to locate OTPS2 dynamic library %s.' % (' or '.join(dllpaths),))
   sys.exit(1)

# Load OTPS2 library.
otps2_ = ctypes.CDLL(str(dllpath))

CONTIGUOUS = str('CONTIGUOUS')

# Access to model objects (variables, parameters, dependencies, couplings, model instances)
otps2_.predict_tide.argtypes = [
    ctypes.c_int,
    numpy.ctypeslib.ndpointer(dtype=ctypes.c_char, ndim=2, flags=CONTIGUOUS),
    numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1, flags=CONTIGUOUS),
    numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1, flags=CONTIGUOUS),
    ctypes.c_double,
    ctypes.c_int,
    numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1, flags=CONTIGUOUS),
    numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1, flags=CONTIGUOUS),
]
otps2_.predict_tide.restype = None

otps2_.predict_tide_2d.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    numpy.ctypeslib.ndpointer(dtype=ctypes.c_char, ndim=2, flags=CONTIGUOUS),
    numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=2, flags=CONTIGUOUS),
    numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=2, flags=CONTIGUOUS),
    numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1, flags=CONTIGUOUS),
    ctypes.c_int,
    numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1, flags=CONTIGUOUS),
    numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=2, flags=CONTIGUOUS),
]
otps2_.predict_tide_2d.restype = None

reference_time = datetime.datetime(1858, 11, 17)

def predict_tide(components, latitude, start_time, ntime, delta_time):
    ncon = len(components)
    z1Re = numpy.empty((ncon,), dtype=float)
    z1Im = numpy.empty((ncon,), dtype=float)
    cid = numpy.full((ncon, 4), ' ', dtype=ctypes.c_char)
    for i, (name, (Re, Im)) in enumerate(components.items()):
        cid[i, :len(name)] = name
        z1Re[i] = Re
        z1Im[i] = Im
    times = (delta_time * numpy.arange(ntime) + (start_time - reference_time).total_seconds()) / 86400.
    result = numpy.empty((ntime,), dtype=float)
    otps2_.predict_tide(ncon, cid, z1Re, z1Im, latitude, ntime, times, result)
    return result

def predict_tide_2d(components, latitude, start_time: datetime.datetime, ntime: int, delta_time: float):
    ncon = len(components)
    latitude = numpy.array(latitude)
    n = latitude.size
    z1Re = numpy.empty((ncon, n), dtype=float)
    z1Im = numpy.empty((ncon, n), dtype=float)
    cid = numpy.full((ncon, 4), ' ', dtype=ctypes.c_char)
    for i, (name, (Re, Im)) in enumerate(components.items()):
        cid[i, :len(name)] = name
        assert Re.shape == latitude.shape
        assert Im.shape == latitude.shape
        z1Re[i, :] = Re
        z1Im[i, :] = Im
    times = (delta_time * numpy.arange(ntime) + (start_time - reference_time).total_seconds()) / 86400.
    result = numpy.empty((ntime,) + latitude.shape, dtype=float)
    otps2_.predict_tide_2d(n, ncon, cid, z1Re, z1Im, latitude, ntime, times, result)
    return result

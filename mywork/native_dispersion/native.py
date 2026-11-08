"""Thin construction binding for the compiled native OpenMM Force; no Python callback."""
import ctypes
import math
from pathlib import Path
import numpy as np

_library = None


def add_correction(system, coefficients, box, cutoff=.6, tolerance=1e-4, group=30, grid=None):
    global _library
    if _library is None:
        _library = ctypes.CDLL(str(Path(__file__).resolve().parent/'build/libNativeDispersion.so'))
        _library.nd_add_force.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
                                         ctypes.POINTER(ctypes.c_int), ctypes.c_double, ctypes.c_double, ctypes.c_int]
        _library.nd_add_force.restype = ctypes.c_int
        _library.nd_last_error.restype = ctypes.c_char_p
    coefficients = np.ascontiguousarray(coefficients, dtype=np.float64)
    box = np.asarray(box, dtype=float)
    if coefficients.shape != (system.getNumParticles(), 3):
        raise ValueError('Expected C6/C8/C10 for every particle, in kJ/mol nm^n')
    if box.shape != (3,3) or not np.allclose(box,np.diag(np.diag(box)),atol=1e-12,rtol=0):
        raise ValueError('Only orthorhombic boxes supported')
    if not 0 < tolerance < .5 or cutoff <= 0:
        raise ValueError('Invalid cutoff/tolerance')
    alpha = math.sqrt(-math.log(2*tolerance))/cutoff
    if grid is None:
        grid = np.ceil(2*alpha*np.diag(box)/3/tolerance**.2).astype(np.int32)
    grid = np.ascontiguousarray(grid,dtype=np.int32)
    if grid.shape != (3,):
        raise ValueError('Grid must have three sizes')
    index = _library.nd_add_force(int(system.this), len(coefficients),
        coefficients.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        grid.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), alpha, cutoff, group)
    if index < 0:
        raise RuntimeError(_library.nd_last_error().decode())
    return dict(force_index=index, group=group, grid=grid.tolist(), alpha_per_nm=alpha,
                cutoff_nm=cutoff, method='native C6/C8/C10 PME minus raw cutoff dispersion')

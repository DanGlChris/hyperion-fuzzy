import ctypes
import sys
import os
import platform
import numpy as np
from ctypes import POINTER, Structure, c_double, c_int, byref
from typing import List, Tuple

# Determine shared library file extension
platform_system = platform.system()
if platform_system == "Windows":
    hypersphere_lib_name = "fuzzy_contribution.dll"
    fuzzy_lib_name = "fuzzy_contribution.dll"
    optimize_lib_name = "optimize_hypersphere.dll"
    test_lib_name = "test.dll"
elif platform_system == "Darwin":
    hypersphere_lib_name = "fuzzy_contribution.dylib"
    fuzzy_lib_name = "fuzzy_contribution.dylib"
    optimize_lib_name = "optimize_hypersphere.dylib"
    test_lib_name = "test.dylib"
elif platform_system == "Linux":
    hypersphere_lib_name = "fuzzy_contribution.so"
    fuzzy_lib_name = "fuzzy_contribution.so"
    optimize_lib_name = "optimize_hypersphere.so"
    test_lib_name = "test.so"
else:
    raise RuntimeError(f"Unsupported OS: {platform_system}")

# Load shared libraries
try:
    hypersphere_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "build", hypersphere_lib_name))
    fuzzy_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "build", fuzzy_lib_name))
    optimize_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "build", optimize_lib_name))
    test_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "build", test_lib_name))
except OSError as e:
    raise RuntimeError(f"Failed to load shared libraries: {e}")


# Define argument and return types
hypersphere_lib.create_hypersphere.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                                   ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
                                   ctypes.c_double]
hypersphere_lib.create_hypersphere.restype = ctypes.c_void_p

hypersphere_lib.delete_hypersphere.argtypes = [ctypes.c_void_p]

hypersphere_lib.set_center.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
hypersphere_lib.get_center.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]

hypersphere_lib.set_radius.argtypes = [ctypes.c_void_p, ctypes.c_double]
hypersphere_lib.get_radius.argtypes = [ctypes.c_void_p]
hypersphere_lib.get_radius.restype = ctypes.c_double

hypersphere_lib.get_ux.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]

hypersphere_lib.add_assignment.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_double]

class Hypersphere:
    def __init__(self, center: np.ndarray, radius: float, initial_elements: np.ndarray):
        center = center.astype(np.float64)
        initial_elements = initial_elements.astype(np.float64)

        self.instance = hypersphere_lib.create_hypersphere(center.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                               center.size,
                                               initial_elements.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                               initial_elements.shape[0], initial_elements.shape[1],
                                               ctypes.c_double(radius))

    def __del__(self):
        hypersphere_lib.delete_hypersphere(self.instance)

    def set_center(self, new_center: np.ndarray):
        new_center = new_center.astype(np.float64)
        hypersphere_lib.set_center(self.instance, new_center.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), new_center.size)

    def get_center(self) -> np.ndarray:
        center = np.zeros(3, dtype=np.float64)
        hypersphere_lib.get_center(self.instance, center.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return center

    def set_radius(self, radius: float):
        hypersphere_lib.set_radius(self.instance, ctypes.c_double(radius))

    def get_radius(self) -> float:
        return hypersphere_lib.get_radius(self.instance)

    def get_ux(self) -> np.ndarray:
        ux = np.zeros(3, dtype=np.float64)
        hypersphere_lib.get_ux(self.instance, ux.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return ux

    def add_assignment(self, array: np.ndarray, value: int, weight: float):
        array = array.astype(np.float64)
        hypersphere_lib.add_assignment(self.instance, array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), array.size, value, weight)

    def clear_assignments(self):
        hypersphere_lib.clear_assignments(self.instance)


# Define C++ function signatures
fuzzy_lib.get_assignments.argtypes = [POINTER(HypersphereC), POINTER(POINTER(Assignment)), POINTER(c_int)]
fuzzy_lib.get_assignments.restype = None

fuzzy_lib.fuzzy_contribution.argtypes = [
    POINTER(c_double),
    POINTER(HypersphereC), POINTER(HypersphereC),
    c_int, c_int, c_int,
    c_double, c_double, c_double,
    POINTER(c_int), POINTER(c_double),
]
fuzzy_lib.fuzzy_contribution.restype = None

optimize_lib.optimize_hypersphere.argtypes = [
    POINTER(HypersphereC), POINTER(HypersphereC), c_int,
    c_double, c_double, c_int, c_double, c_int,
]
optimize_lib.optimize_hypersphere.restype = None

# Python Hypersphere class
class Hypersphere:
    def __init__(self, center: List[float], radius: float, initial_elements: List[List[float]]):
        self.dim = len(center)
        self.center = np.array(center, dtype=np.float64)
        self.radius = float(radius)
        self.initial_elements = np.array(initial_elements, dtype=np.float64)
        self.ux = np.mean(self.initial_elements, axis=0) if len(self.initial_elements) > 0 else np.zeros(self.dim)
        self.assignments = []

        self.c_hypersphere = HypersphereC(
            initial_elements=self.initial_elements.ctypes.data_as(POINTER(c_double)),
            center=self.center.ctypes.data_as(POINTER(c_double)),
            ux=self.ux.ctypes.data_as(POINTER(c_double)),
            radius=self.radius,
            num_elements=len(self.initial_elements),
        )

    def get_c_structure(self):
        return byref(self.c_hypersphere)

    def get_assignments(self):
        output_ptr = POINTER(Assignment)()
        output_size = c_int()
        fuzzy_lib.get_assignments(self.get_c_structure(), byref(output_ptr), byref(output_size))

        self.assignments = []
        for i in range(output_size.value):
            assignment = output_ptr[i]
            x_array = np.ctypeslib.as_array(assignment.x, shape=(self.dim,))
            self.assignments.append({"x": x_array.tolist(), "label": assignment.label, "value": assignment.value})

        return self.assignments

# Static functions
def fuzzy_contribution(x: List[float], pos_hypers: List[Hypersphere], neg_hypers: List[Hypersphere], gamma: float, sigma: float, E: float):
    x_np = np.array(x, dtype=np.float64)
    assigned_class = c_int()
    contribution = c_double()

    fuzzy_lib.fuzzy_contribution(
        x_np.ctypes.data_as(POINTER(c_double)),
        (HypersphereC * len(pos_hypers))(*[hs.c_hypersphere for hs in pos_hypers]),
        (HypersphereC * len(neg_hypers))(*[hs.c_hypersphere for hs in neg_hypers]),
        len(pos_hypers), len(neg_hypers), len(x),
        gamma, sigma, E,
        byref(assigned_class), byref(contribution),
    )

    return assigned_class.value, contribution.value

def optimize_hypersphere(hypersphere: Hypersphere, others: List[Hypersphere], c1: float, learning_rate: float = 0.01, max_iterations: int = 100, tolerance: float = 1e-6):
    optimize_lib.optimize_hypersphere(
        byref(hypersphere.c_hypersphere),
        (HypersphereC * len(others))(*[hs.c_hypersphere for hs in others]),
        len(others),
        c1, learning_rate, max_iterations, tolerance, hypersphere.dim,
    )

def predict(data: np.ndarray, pos_hypers: List[Hypersphere], neg_hypers: List[Hypersphere], sigma: float) -> List[int]:
    num_samples, dim = data.shape
    predictions = np.zeros(num_samples, dtype=np.int32)

    fuzzy_lib.predict(
        data.ctypes.data_as(POINTER(c_double)),
        num_samples, dim,
        (HypersphereC * len(pos_hypers))(*[hs.c_hypersphere for hs in pos_hypers]),
        len(pos_hypers),
        (HypersphereC * len(neg_hypers))(*[hs.c_hypersphere for hs in neg_hypers]),
        len(neg_hypers),
        sigma,
        predictions.ctypes.data_as(POINTER(c_int)),
    )

    return predictions.tolist()
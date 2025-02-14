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

hypersphere_lib.clear_assignments.argtypes = [ctypes.c_void_p]

# New functions to access initialElements and assignments
hypersphere_lib.get_num_initial_elements.argtypes = [ctypes.c_void_p]
hypersphere_lib.get_num_initial_elements.restype = ctypes.c_int

hypersphere_lib.get_initial_elements.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]

hypersphere_lib.get_num_assignments.argtypes = [ctypes.c_void_p]
hypersphere_lib.get_num_assignments.restype = ctypes.c_int

hypersphere_lib.get_assignments.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)]


# Define argument and return types for optimize_hypersphere function
optimize_lib.optimize_hypersphere.argtypes = [
    ctypes.c_void_p,  # Target hypersphere
    ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,  # Other hyperspheres + count
    ctypes.c_double,  # c1
    ctypes.c_double,  # learning_rate
    ctypes.c_int,  # max_iterations
    ctypes.c_double,  # tolerance
    ctypes.c_int  # dim
]

class Hypersphere:
    def __init__(self, center: np.ndarray, radius: float, initial_elements: np.ndarray):
        center = center.astype(np.float64)
        initial_elements = initial_elements.astype(np.float64)

        self.instance = hypersphere_lib.create_hypersphere(center.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                               center.size,
                                               initial_elements.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                               initial_elements.shape[0], initial_elements.shape[1],
                                               ctypes.c_double(radius))
        
        self.center = center
        self.radius = radius
        self.assignments = []
        self.element_size = initial_elements.shape[1]

    def __del__(self):
        hypersphere_lib.delete_hypersphere(self.instance)

    def set_center(self, new_center: np.ndarray):
        new_center = new_center.astype(np.float64)
        hypersphere_lib.set_center(self.instance, new_center.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), new_center.size)
        self.center = new_center  # Sync with C++ memory

    def get_center(self) -> np.ndarray:
        center = np.zeros(3, dtype=np.float64)
        hypersphere_lib.get_center(self.instance, center.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        self.center = center  # Sync Python object with C++ memory
        return center

    def set_radius(self, radius: float):
        hypersphere_lib.set_radius(self.instance, ctypes.c_double(radius))
        self.radius = radius  # Sync with C++ memory

    def get_radius(self) -> float:
        self.radius = hypersphere_lib.get_radius(self.instance)
        return self.radius

    def get_ux(self) -> np.ndarray:
        ux = np.zeros(3, dtype=np.float64)
        hypersphere_lib.get_ux(self.instance, ux.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return ux

    def add_assignment(self, array: np.ndarray, value: int, weight: float):
        array = array.astype(np.float64)
        hypersphere_lib.add_assignment(
            self.instance, array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), array.size, value, weight
        )
        self.get_assignments()  # Sync Python object with C++ memory

    def clear_assignments(self):
        hypersphere_lib.clear_assignments(self.instance)
        self.assignments = []  # Reset Python-side cache


    def get_initial_elements(self) -> np.ndarray:
        """Retrieve the initial elements from the hypersphere."""
        num_elements = hypersphere_lib.get_num_initial_elements(self.instance)
        if num_elements == 0:
            return np.array([])

        elements = np.zeros((num_elements, self.element_size), dtype=np.float64)
        hypersphere_lib.get_initial_elements(self.instance, elements.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return elements

    def get_assignments(self):
        """Retrieve all assignments from the hypersphere."""
        num_assignments = hypersphere_lib.get_num_assignments(self.instance)
        if num_assignments == 0:
            return []

        assignment_arrays = np.zeros((num_assignments, self.element_size), dtype=np.float64)
        assignment_values = np.zeros(num_assignments, dtype=np.int32)
        assignment_weights = np.zeros(num_assignments, dtype=np.float64)

        hypersphere_lib.get_assignments(
            self.instance,
            assignment_arrays.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            assignment_values.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            assignment_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )

        self.assignments = [(assignment_arrays[i], assignment_values[i], assignment_weights[i]) for i in range(num_assignments)]
        return self.assignments
    
    def optimize(self, other_hyperspheres: list, c1: float, learning_rate: float, max_iterations: int, tolerance: float):
        """Optimize the hypersphere using Dlib and update Python-side values"""
        num_other = len(other_hyperspheres)
        other_hypersphere_ptrs = (ctypes.c_void_p * num_other)(*[hs.instance for hs in other_hyperspheres])

        optimize_lib.optimize_hypersphere(
            self.instance,
            other_hypersphere_ptrs, num_other,
            ctypes.c_double(c1),
            ctypes.c_double(learning_rate),
            ctypes.c_int(max_iterations),
            ctypes.c_double(tolerance),
            ctypes.c_int(self.dim)
        )

        self.get_center()  # Fetch updated center from C++
        self.get_radius()  # Fetch updated radius from C++

# Define argument and return types for fuzzy_contribution functions
fuzzy_lib.fuzzy_contribution.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # x
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),  # positive & negative hyperspheres
    ctypes.c_int, ctypes.c_int, ctypes.c_int,  # num_positive, num_negative, dim
    ctypes.c_double, ctypes.c_double, ctypes.c_double,  # gamma, sigma, E
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)  # assigned_class, contribution
]

fuzzy_lib.predict.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,  # transformed_data, num_samples, dim
    ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,  # positive hyperspheres & count
    ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,  # negative hyperspheres & count
    ctypes.c_double, ctypes.POINTER(ctypes.c_int)  # sigma, predictions (output)
]


def fuzzy_contribution(x: np.ndarray, positive_hyperspheres: list, negative_hyperspheres: list, gamma: float, sigma: float, E: float):
    x = x.astype(np.float64)
    dim = x.size

    num_positive = len(positive_hyperspheres)
    num_negative = len(negative_hyperspheres)

    positive_hypersphere_ptrs = (ctypes.c_void_p * num_positive)(*[hs.instance for hs in positive_hyperspheres])
    negative_hypersphere_ptrs = (ctypes.c_void_p * num_negative)(*[hs.instance for hs in negative_hyperspheres])

    assigned_class = ctypes.c_int()
    contribution = ctypes.c_double()

    fuzzy_lib.fuzzy_contribution(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        positive_hypersphere_ptrs, negative_hypersphere_ptrs,
        num_positive, num_negative, dim,
        ctypes.c_double(gamma), ctypes.c_double(sigma), ctypes.c_double(E),
        ctypes.byref(assigned_class), ctypes.byref(contribution)
    )

    # Retrieve updated assignments
    for hs in positive_hyperspheres + negative_hyperspheres:
        hs.get_assignments()  # This syncs the Python object with C++ memory

    return assigned_class.value, contribution.value


def predict(transformed_data: np.ndarray, positive_hyperspheres: list, negative_hyperspheres: list, sigma: float):
    """Predict class labels for transformed data using hyperspheres."""
    transformed_data = transformed_data.astype(np.float64)
    num_samples, dim = transformed_data.shape

    num_positive = len(positive_hyperspheres)
    num_negative = len(negative_hyperspheres)

    # Convert Hypersphere Python objects to C++ pointers
    positive_hypersphere_ptrs = (ctypes.c_void_p * num_positive)(*[hs.instance for hs in positive_hyperspheres])
    negative_hypersphere_ptrs = (ctypes.c_void_p * num_negative)(*[hs.instance for hs in negative_hyperspheres])

    # Allocate memory for the output predictions
    predictions = np.zeros(num_samples, dtype=np.int32)

    # Call the C++ function
    fuzzy_lib.predict(
        transformed_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_int(num_samples), ctypes.c_int(dim),
        positive_hypersphere_ptrs, ctypes.c_int(num_positive),
        negative_hypersphere_ptrs, ctypes.c_int(num_negative),
        ctypes.c_double(sigma), predictions.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )

    return predictions

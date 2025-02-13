import ctypes
import sys
import os
import platform
from typing import Tuple
import numpy as np

# Determine the appropriate file extension based on the operating system
platform_system = platform.system()

if platform_system == 'Windows':  # Windows
    fuzzy_lib_name = "fuzzy_contribution.dll"
    optimize_lib_name = "optimize_hypersphere.dll"
    test_lib_name = "test.dll"
elif platform_system == 'Darwin':  # macOS
    fuzzy_lib_name = ".fuzzy_contribution.dylib"
    optimize_lib_name = "optimize_hypersphere.dylib"
    test_lib_name = "test.dylib"
elif platform_system == "Linux":  # Assume Linux or other Unix-like OS
    fuzzy_lib_name = "fuzzy_contribution.so"
    optimize_lib_name = "optimize_hypersphere.so"
    test_lib_name = "test.so"
else:
    print("Unsupported operating system:", platform_system)
    sys.exit(1)

# Load compiled shared libraries
try:
    fuzzy_lib_path = os.path.join(os.path.dirname(__file__), 'build', fuzzy_lib_name)
    optimize_lib_path = os.path.join(os.path.dirname(__file__), 'build', optimize_lib_name)
    test_lib_path = os.path.join(os.path.dirname(__file__), 'build', test_lib_name)
    
    test_lib = ctypes.CDLL(test_lib_path)
    fuzzy_lib = ctypes.CDLL(fuzzy_lib_path)
    optimize_lib = ctypes.CDLL(optimize_lib_path)
except OSError as e:
    raise RuntimeError(f"Failed to load shared libraries: {e}")

# Define the Hypersphere structure
class Hypersphere(ctypes.Structure):
    _fields_ = [
        ("initial_elements", ctypes.POINTER(ctypes.c_double)),
        ("center", ctypes.POINTER(ctypes.c_double)),
        ("ux", ctypes.POINTER(ctypes.c_double)),
        ("radius", ctypes.c_double),
        ("num_elements", ctypes.c_int),
        #("assignments", ctypes.POINTER(ctypes.c_void_p))  # Placeholder for assignments
    ]

# Define the argument and return types for the euclideanDistance function
test_lib.euclideanDistance.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_size_t)
test_lib.euclideanDistance.restype = ctypes.c_double

def euclidean_distance(point1, point2):
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimension")
    
    point1_array = (ctypes.c_double * len(point1))(*point1)
    point2_array = (ctypes.c_double * len(point2))(*point2)
    
    return test_lib.euclideanDistance(point1_array, point2_array, len(point1))

print("test")
# Define the function signatures for the shared library
fuzzy_lib.fuzzy_contribution.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # x
    ctypes.POINTER(Hypersphere),      # positive_hyperspheres
    ctypes.POINTER(Hypersphere),      # negative_hyperspheres
    ctypes.c_int,                     # num_positive
    ctypes.c_int,                     # num_negative
    ctypes.c_int,                     # dim
    ctypes.c_double,                  # gamma
    ctypes.c_double,                  # sigma
    ctypes.c_double,                  # E
    ctypes.POINTER(ctypes.c_int),     # assigned_class
    ctypes.POINTER(ctypes.c_double)   # contribution
]
fuzzy_lib.fuzzy_contribution.restype = None

fuzzy_lib.predict.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # transformed_data
    ctypes.c_int,                     # num_samples
    ctypes.c_int,                     # dim
    ctypes.POINTER(Hypersphere),      # positive_hyperspheres
    ctypes.c_int,                     # num_positive
    ctypes.POINTER(Hypersphere),      # negative_hyperspheres
    ctypes.c_int,                     # num_negative
    ctypes.c_double,                  # sigma
    ctypes.POINTER(ctypes.c_int)      # predictions
]
fuzzy_lib.predict.restype = None

# Expose the get_assignments function
fuzzy_lib.get_assignments.argtypes = [ctypes.POINTER(Hypersphere), ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.POINTER(ctypes.c_int)]
fuzzy_lib.get_assignments.restype = None

# Function to retrieve assignments from a Hypersphere
def get_assignments(hypersphere):
    output_ptr = ctypes.POINTER(ctypes.c_double)()
    output_size = ctypes.c_int()

    fuzzy_lib.get_assignments(ctypes.byref(hypersphere), ctypes.byref(output_ptr), ctypes.byref(output_size))

    # Convert the output to a NumPy array
    assignments = np.ctypeslib.as_array(output_ptr, shape=(output_size.value,))
    
    if platform_system == "Windows":
        # Windows-specific memory deallocation
        ctypes.windll.kernel32.HeapFree(ctypes.windll.kernel32.GetProcessHeap(), 0, output_ptr)
    elif platform_system == "Darwin":
        # macOS-specific memory deallocation
        # In macOS, you typically use `free()` from the C standard library
        libc = ctypes.CDLL("libc.dylib")
        libc.free(output_ptr)
    elif platform_system == "Linux":
        # Linux-specific memory deallocation
        # In Linux, you also typically use `free()` from the C standard library
        libc = ctypes.CDLL("libc.so.6")
        libc.free(output_ptr)
    else:
        print("Unsupported operating system:", platform.system())
        sys.exit(1)

    return assignments

def predict(transformed_data, positive_hyperspheres, negative_hyperspheres, sigma):
    transformed_data = np.asarray(transformed_data, dtype=np.float64)
    num_samples, dim = transformed_data.shape

    positive_hyperspheres_c = (Hypersphere * len(positive_hyperspheres))(
        *[Hypersphere(
            initial_elements=np.asarray(h.initial_elements, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            center=np.asarray(h.center, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ux=np.asarray(h.ux, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            radius=h.radius,
            num_elements=len(h.initial_elements)
        ) for h in positive_hyperspheres]
    )

    negative_hyperspheres_c = (Hypersphere * len(negative_hyperspheres))(
        *[Hypersphere(
            initial_elements=np.asarray(h.initial_elements, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            center=np.asarray(h.center, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ux=np.asarray(h.ux, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            radius=h.radius,
            num_elements=len(h.initial_elements)
        ) for h in negative_hyperspheres]
    )

    predictions = np.zeros(num_samples, dtype=np.int32)

    fuzzy_lib.predict(
        transformed_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(num_samples),
        ctypes.c_int(dim),
        positive_hyperspheres_c,
        ctypes.c_int(len(positive_hyperspheres)),
        negative_hyperspheres_c,
        ctypes.c_int(len(negative_hyperspheres)),
        ctypes.c_double(sigma),
        predictions.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )

    return predictions.tolist()

def fuzzy_contribution(x, positive_hyperspheres, negative_hyperspheres, assignments, gamma, sigma, E):
    x = np.asarray(x, dtype=np.float64)
    num_positive = len(positive_hyperspheres)
    num_negative = len(negative_hyperspheres)
    dim = len(x)

    # Convert positive hyperspheres to ctypes structures
    positive_hyperspheres_c = (Hypersphere * num_positive)(
        *[Hypersphere(
            initial_elements=np.asarray(h.initial_elements, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            center=np.asarray(h.center, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ux=np.asarray(h.ux, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            radius=h.radius,
            num_elements=len(h.initial_elements)
        ) for h in positive_hyperspheres]
    )

    # Convert negative hyperspheres to ctypes structures
    negative_hyperspheres_c = (Hypersphere * num_negative)(
        *[Hypersphere(
            initial_elements=np.asarray(h.initial_elements, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            center=np.asarray(h.center, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ux=np.asarray(h.ux, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            radius=h.radius,
            num_elements=len(h.initial_elements)
        ) for h in negative_hyperspheres]
    )

    assigned_class = ctypes.c_int()
    contribution = ctypes.c_double()

    # Call the C++ function
    fuzzy_lib.fuzzy_contribution(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        positive_hyperspheres_c,
        negative_hyperspheres_c,
        ctypes.c_int(num_positive),
        ctypes.c_int(num_negative),
        ctypes.c_int(dim),
        ctypes.c_double(gamma),
        ctypes.c_double(sigma),
        ctypes.c_double(E),
        ctypes.byref(assigned_class),
        ctypes.byref(contribution)
    )

    # Convert NumPy array `x` to a tuple for consistency in assignments
    x_tuple = tuple(x.tolist())

    # Store the assignment in the Python list instead of modifying Hypersphere
    if assigned_class.value == 1:
        assignments.append((x_tuple, 1, contribution.value))
    elif assigned_class.value == -1:
        assignments.append((x_tuple, -1, contribution.value))
    else:
        assignments.append((x_tuple, 0, 1))  # Noise

# ----- Optimize Hypersphere Library Setup -----

# Define the argument and return types for the function
optimize_lib.optimize_hypersphere.argtypes = [
    ctypes.POINTER(Hypersphere),      # hypersphere
    ctypes.POINTER(Hypersphere),      # other_hyperspheres
    ctypes.c_double,                  # c1
    ctypes.c_double,                  # learning_rate
    ctypes.c_int,                     # max_iterations
    ctypes.c_double,                  # tolerance
    ctypes.c_int                      # dim
]
optimize_lib.optimize_hypersphere.restype = None

def optimize_hypersphere(self, hypersphere, other_hyperspheres, c1, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
    hypersphere_c = Hypersphere(
        initial_elements=np.asarray(hypersphere.initial_elements, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        center=np.asarray(hypersphere.center, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ux=np.asarray(hypersphere.ux, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        radius=hypersphere.radius,
        num_elements=len(hypersphere.initial_elements),
        assignments=None  # Placeholder for assignments
    )

    other_hyperspheres_c = (Hypersphere * len(other_hyperspheres))(
        *[Hypersphere(
            initial_elements=np.asarray(h.initial_elements, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            center=np.asarray(h.center, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ux=np.asarray(h.ux, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            radius=h.radius,
            num_elements=len(h.initial_elements),
            assignments=None  # Placeholder for assignments
        ) for h in other_hyperspheres]
    )

    optimize_lib.optimize_hypersphere(
        ctypes.byref(hypersphere_c),
        other_hyperspheres_c,
        ctypes.c_double(c1),
        ctypes.c_double(learning_rate),
        ctypes.c_int(max_iterations),
        ctypes.c_double(tolerance),
        ctypes.c_int(len(hypersphere.center))
    )

    hypersphere.radius = hypersphere_c.radius
    hypersphere.center = np.ctypeslib.as_array(hypersphere_c.center, shape=(len(hypersphere.center),))
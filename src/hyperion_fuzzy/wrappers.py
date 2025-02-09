import ctypes
from typing import Tuple
import numpy as np

# Load compiled shared libraries
try:
    fuzzy_lib = ctypes.CDLL("../build/fuzzy_contribution.so")
    optimize_lib = ctypes.CDLL("../build/optimize_hypersphere.so")
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
        ("assignments", ctypes.POINTER(ctypes.c_void_p))  # Placeholder for assignments
    ]

# Define the argument and return types for the function
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

    positive_hyperspheres_c = (Hypersphere * num_positive)(
        *[Hypersphere(
            initial_elements=np.asarray(h.initial_elements, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            center=np.asarray(h.center, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ux=np.asarray(h.ux, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            radius=h.radius,
            num_elements=len(h.initial_elements)
        ) for h in positive_hyperspheres]
    )

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

    if assigned_class.value == 1:
        assignm = (x, 1, contribution.value)
        assignments.append(assignm)
        positive_hyperspheres[0].assignments.append(assignm)
    elif assigned_class.value == -1:
        assignm = (x, -1, contribution.value)
        assignments.append(assignm)
        negative_hyperspheres[0].assignments.append(assignm)
    else:
        assignments.append((x, 0, 1))  # Noise

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
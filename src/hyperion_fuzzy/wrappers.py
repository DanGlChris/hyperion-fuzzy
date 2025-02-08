import ctypes
import numpy as np

# Load shared libraries
distance_lib = ctypes.CDLL('./build/distance.so')
fuzzy_contribution_lib = ctypes.CDLL('./build/fuzzy_contribution.so')
optimize_hypersphere_lib = ctypes.CDLL('./build/optimize_hypersphere.so')

# Define distance functions
distance_lib.compute_distances.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.c_size_t, ctypes.c_size_t]
distance_lib.compute_distances.restype = ctypes.POINTER(ctypes.c_double)

def compute_distances(x, points):
    x_arr = np.array(x, dtype=np.float64)
    points_arr = np.array(points, dtype=np.float64)
    num_points = points_arr.shape[0]
    dim = points_arr.shape[1]

    points_ptr = (ctypes.POINTER(ctypes.c_double) * num_points)()
    for i in range(num_points):
        points_ptr[i] = points_arr[i, :].ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    result_ptr = distance_lib.compute_distances(
        x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        points_ptr,
        num_points,
        dim
    )

    result = np.ctypeslib.as_array(result_ptr, shape=(num_points,))
    return result

## Define argument and return types for fuzzy contribution function
fuzzy_contribution_lib.fuzzy_contribution.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # x
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # positive_centers
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # negative_centers
    ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),  # positive_elements
    ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),  # negative_elements
    ctypes.c_double,  # c1
    ctypes.c_double,  # sigma
    ctypes.c_double,  # epsilon
    ctypes.POINTER(ctypes.c_int),  # label (output)
    ctypes.POINTER(ctypes.c_double)  # contribution (output)
]
fuzzy_contribution_lib.fuzzy_contribution.restype = None

def fuzzy_contribution_cpp(x, positive_centers, negative_centers, positive_elements, negative_elements, c1, sigma, epsilon):
    # Convert inputs to ctypes
    x_arr = np.array(x, dtype=np.float64)
    positive_centers_arr = np.array(positive_centers, dtype=np.float64)
    negative_centers_arr = np.array(negative_centers, dtype=np.float64)

    # Convert elements to 3D ctypes arrays
    positive_elements_c = (ctypes.POINTER(ctypes.POINTER(ctypes.c_double)) * len(positive_elements))()
    for i, group in enumerate(positive_elements):
        positive_elements_c[i] = (ctypes.POINTER(ctypes.c_double) * len(group))()
        for j, el in enumerate(group):
            positive_elements_c[i][j] = el.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    negative_elements_c = (ctypes.POINTER(ctypes.POINTER(ctypes.c_double)) * len(negative_elements))()
    for i, group in enumerate(negative_elements):
        negative_elements_c[i] = (ctypes.POINTER(ctypes.c_double) * len(group))()
        for j, el in enumerate(group):
            negative_elements_c[i][j] = el.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # Prepare output variables
    label = ctypes.c_int()
    contribution = ctypes.c_double()

    # Call the C++ function
    fuzzy_contribution_lib.fuzzy_contribution(
        x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        positive_centers_arr.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
        negative_centers_arr.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
        positive_elements_c,
        negative_elements_c,
        c1,
        sigma,
        epsilon,
        ctypes.byref(label),
        ctypes.byref(contribution)
    )

    return label.value, contribution.value

# Define conformal kernel wrapper
fuzzy_contribution_lib.conformal_kernel.argtypes = [...]
fuzzy_contribution_lib.conformal_kernel.restype = ctypes.c_double

def conformal_kernel_cpp(...):
    pass  # Define the wrapper similarly


# Define argument and return types for the C++ optimize_hypersphere function
optimize_hypersphere_lib.optimize_hypersphere.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # initial_params
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # elements
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # other_hyperspheres
    ctypes.c_double,  # c1
    ctypes.c_double,  # learning_rate
    ctypes.c_int      # max_iters
]
optimize_hypersphere_lib.optimize_hypersphere.restype = ctypes.POINTER(ctypes.c_double)

def optimize_hypersphere_cpp(initial_params, elements, other_hyperspheres, c1, learning_rate=0.01, max_iters=100):
    # Convert inputs to ctypes
    initial_params_arr = np.array(initial_params, dtype=np.float64)
    elements_arr = np.array(elements, dtype=np.float64)
    other_hyperspheres_arr = np.array(other_hyperspheres, dtype=np.float64)

    # Convert to ctypes pointers
    elements_ptr = (ctypes.POINTER(ctypes.c_double) * len(elements_arr))()
    for i, elem in enumerate(elements_arr):
        elements_ptr[i] = elem.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    other_hyperspheres_ptr = (ctypes.POINTER(ctypes.c_double) * len(other_hyperspheres_arr))()
    for i, other in enumerate(other_hyperspheres_arr):
        other_hyperspheres_ptr[i] = other.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # Call the C++ function
    result_ptr = optimize_hypersphere_lib.optimize_hypersphere(
        initial_params_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        elements_ptr,
        other_hyperspheres_ptr,
        c1,
        learning_rate,
        max_iters
    )

    # Convert result back to numpy array
    result = np.ctypeslib.as_array(result_ptr, shape=(len(initial_params),))
    return result
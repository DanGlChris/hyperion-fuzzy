import numpy as np

# Import the compiled pybind11 modules
import hypersphere_module  # The module from `hypersphere_bindings.cpp`
import optimize_module  # The module from `optimize_bindings.cpp`
import fuzzy_module  # The module from `fuzzy_bindings.cpp`


class Hypersphere:
    def __init__(self, center: np.ndarray, radius: float, initial_elements: np.ndarray):
        """Create a new Hypersphere using pybind11 bindings."""
        self.instance = hypersphere_module.Hypersphere(center.tolist(), radius, initial_elements.tolist())
        self.center = center
        self.radius = radius
        self.assignments = []

    def set_center(self, new_center: np.ndarray):
        self.instance.set_center(new_center.tolist())
        self.center = new_center  # Sync Python attribute

    def get_center(self) -> np.ndarray:
        self.center = np.array(self.instance.get_center())  # Sync Python attribute
        return self.center

    def set_radius(self, radius: float):
        self.instance.set_radius(radius)
        self.radius = radius  # Sync Python attribute

    def get_radius(self) -> float:
        self.radius = self.instance.get_radius()  # Sync Python attribute
        return self.radius

    def get_ux(self) -> np.ndarray:
        return np.array(self.instance.get_ux())

    def add_assignment(self, array: np.ndarray, value: int, weight: float):
        self.instance.add_assignment(array.tolist(), value, weight)

    def clear_assignments(self):
        self.instance.clear_assignments()

    def get_initial_elements(self) -> np.ndarray:
        return np.array(self.instance.get_initial_elements())

    def get_assignments(self):
        """Ensure assignments are always up-to-date from C++."""
        self.assignments = self.instance.get_assignments()
        return self.assignments

    def optimize(self, other_hyperspheres: list, c1: float, learning_rate: float, max_iterations: int, tolerance: float):
        """Optimize the hypersphere using the pybind11 binding and sync updates."""
        self.instance.optimize(
            [hs.instance for hs in other_hyperspheres], c1, learning_rate, max_iterations, tolerance
        )
        # Fetch updated values from C++ and update Python attributes
        self.center = np.array(self.instance.get_center())  
        self.radius = self.instance.get_radius()  

def fuzzy_contribution(x: np.ndarray, positive_hyperspheres: list, negative_hyperspheres: list, gamma: float, sigma: float, E: float):
    """
    Compute fuzzy contribution using pybind11 bindings.
    """
    assigned_class, contribution = fuzzy_module.fuzzy_contribution(
        x.tolist(),
        [hs.instance for hs in positive_hyperspheres],
        [hs.instance for hs in negative_hyperspheres],
        gamma, sigma, E
    )

    # Fetch updated assignments from C++ and sync in Python
    for hs in positive_hyperspheres + negative_hyperspheres:
        hs.assignments = hs.instance.get_assignments()  # Sync Python object with C++ memory

    return assigned_class, contribution


def predict(transformed_data: np.ndarray, positive_hyperspheres: list, negative_hyperspheres: list, sigma: float):
    """
    Predict class labels for transformed data using hyperspheres.
    """
    return fuzzy_module.predict(
        transformed_data.tolist(),
        [hs.instance for hs in positive_hyperspheres],
        [hs.instance for hs in negative_hyperspheres],
        sigma
    )
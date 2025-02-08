import numpy as np
import pandas as pd
from hyperion_fuzzy.wrappers import compute_distances, conformal_kernel_cpp, fuzzy_contribution_cpp, optimize_hypersphere_cpp

class Hypersphere:
    def __init__(self, center, radius, initial_elements):
        self.center = center
        self.radius = radius        
        self.initial_elements = initial_elements
        self.elements = []
        self.assignments = []
        self.ux = (1 / len(self.initial_elements)) * np.sum(self.initial_elements, axis=0)

class HyperionFuzzy:
    def __init__(self, num_clusters=2, c1=1.0, sigma=0.005, E=1e-7, max_iterations=100):
        self.num_clusters = num_clusters
        self.c1 = c1
        self.sigma = sigma
        self.E = E
        self.max_iterations = max_iterations
        self.positive_hyperspheres = []
        self.negative_hyperspheres = []

    def train(self, data, labels):
        transformed_data = data.applymap(self.polynomial_mapping)
        self.positive_hyperspheres, self.negative_hyperspheres = self.initialize_hyperspheres(transformed_data, labels)

        for iteration in range(self.max_iterations):
            assignments = self.fuzzy(transformed_data)

            if all(len(hypersphere.elements) == 0 for hypersphere in self.positive_hyperspheres) or \
               all(len(hypersphere.elements) == 0 for hypersphere in self.negative_hyperspheres):
                break

        return assignments

    def predict(self, new_data):
        transformed_data = new_data.applymap(self.polynomial_mapping)
        predictions = []

        for x in transformed_data.values:
            memberships_p = [conformal_kernel_cpp(x, hs.center, hs.initial_elements, self.sigma, self.E) 
                             for hs in self.positive_hyperspheres]
            memberships_n = [conformal_kernel_cpp(x, hs.center, hs.initial_elements, self.sigma, self.E) 
                             for hs in self.negative_hyperspheres]

            if max(memberships_p) > max(memberships_n):
                predictions.append(1)
            elif max(memberships_n) > max(memberships_p):
                predictions.append(-1)
            else:
                predictions.append(0)

        return np.array(predictions)

    def initialize_hyperspheres(self, data, labels):

        class_p = data[labels == 1]
        class_n = data[labels == -1]

        if len(class_p) < self.num_clusters or len(class_n) < self.num_clusters:
            raise ValueError("Not enough data points to initialize the specified number of clusters.")

        positive_hyperspheres = []
        negative_hyperspheres = []

        for _ in range(self.num_clusters):
            random_point_p = class_p.iloc[np.random.choice(class_p.shape[0])]
            random_point_n = class_n.iloc[np.random.choice(class_n.shape[0])]

            radius = np.linalg.norm(random_point_p - random_point_n) / 3

            positive_hyperspheres.append(Hypersphere(random_point_p.values, radius, class_p.values))
            negative_hyperspheres.append(Hypersphere(random_point_n.values, radius, class_n.values))

        return positive_hyperspheres, negative_hyperspheres

    def polynomial_mapping(self, x):
        return np.exp(x)

    def fuzzy(self, data):

        assignments = []

        for x in data.values:
            label, contribution = fuzzy_contribution_cpp(
                x, 
                [hs.center for hs in self.positive_hyperspheres],
                [hs.center for hs in self.negative_hyperspheres],
                [hs.initial_elements for hs in self.positive_hyperspheres],
                [hs.initial_elements for hs in self.negative_hyperspheres],
                self.c1,
                self.sigma,
                self.E
            )
            assignments.append((x, label, contribution))

        # Optimize hyperspheres using C++ implementation
        for pos_hs, neg_hs in zip(self.positive_hyperspheres, self.negative_hyperspheres):
            self.optimize_hypersphere(pos_hs, self.negative_hyperspheres)
            self.optimize_hypersphere(neg_hs, self.positive_hyperspheres)

        return assignments
    
    def optimize_hypersphere(self, hypersphere, other_hyperspheres):

        initial_params = [hypersphere.radius] + list(hypersphere.center)
        
        # Collect centers of other hyperspheres
        other_centers = [hs.center for hs in other_hyperspheres]

        # Optimize using the C++ function
        optimized_params = optimize_hypersphere_cpp(initial_params, hypersphere.initial_elements, other_centers, self.c1)

        # Update the hypersphere properties
        hypersphere.radius = optimized_params[0]
        hypersphere.center = optimized_params[1:]
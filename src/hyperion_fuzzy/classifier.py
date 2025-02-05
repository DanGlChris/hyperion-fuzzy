import numpy as np
import pandas as pd
from scipy.optimize import minimize
from memory_profiler import memory_usage

import time

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
        start_time = time.time()
        mem_usage_before = memory_usage()[0]
        
        transformed_data = data.applymap(self.polynomial_mapping)  # Transform data
        
        self.positive_hyperspheres, self.negative_hyperspheres = self.initialize_hyperspheres(transformed_data, labels)

        for iteration in range(self.max_iterations):
            assignments = self.fuzzy(transformed_data)

            if all(len(hypersphere.elements) == 0 for hypersphere in self.positive_hyperspheres) or \
               all(len(hypersphere.elements) == 0 for hypersphere in self.negative_hyperspheres):
                break

        end_time = time.time()
        mem_usage_after = memory_usage()[0]

        print(f"Training Time: {end_time - start_time:.4f} seconds")
        print(f"Memory Usage: {mem_usage_after - mem_usage_before:.4f} MiB")

        return assignments

    def predict(self, new_data):
        transformed_data = new_data.applymap(self.polynomial_mapping)  # Transform new data
        predictions = []

        for x in transformed_data.values:
            memberships_p = [self.conformal_kernel(x, hs.center, hs) for hs in self.positive_hyperspheres]
            memberships_n = [self.conformal_kernel(x, hs.center, hs) for hs in self.negative_hyperspheres]

            if max(memberships_p) > max(memberships_n):
                predictions.append(1)  # Positive class
            elif max(memberships_n) > max(memberships_p):
                predictions.append(-1)  # Negative class
            else:
                predictions.append(0)  # Noise or uncertain

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
            #print(f"radius : {radius}")
            
            positive_hyperspheres.append(Hypersphere(random_point_p.values, radius, class_p.values))
            negative_hyperspheres.append(Hypersphere(random_point_n.values, radius, class_n.values))

        return positive_hyperspheres, negative_hyperspheres

    def polynomial_mapping(self, x):
        return np.exp(x)

    def rbf_kernel(self, x, x_prime, sigma):
        return np.exp(-np.linalg.norm(x - x_prime)**2 / (2 * sigma**2))

    def G(self, x, hypersphere):
        distances = np.linalg.norm(hypersphere.initial_elements - x, axis=1)
        ux_distances = np.linalg.norm(hypersphere.ux - hypersphere.initial_elements, axis=1)    
        return np.sum(np.exp(-distances**2 / (ux_distances**2 + self.E)))

    def conformal_kernel(self, x, x_prime, hypersphere, sigma=1.0):
        return self.G(x, hypersphere) * self.rbf_kernel(x, x_prime, sigma) * self.G(x_prime, hypersphere)

    def fuzzy_contribution(self, x, positive_hyperspheres, negative_hyperspheres, assignments, gamma, sigma):
        # Initialize minimum values and corresponding hyperspheres
        min_value_p = float('inf')
        assigned_hypersphere_p = None
        class_label_p = None
        
        min_value_n = float('inf')
        assigned_hypersphere_n = None
        class_label_n = None
        
        # Calculate memberships and find minimums in a single pass
        for hypersphere in positive_hyperspheres:
            k_p = self.conformal_kernel(x, hypersphere.center, hypersphere, sigma)
            if k_p < min_value_p:
                min_value_p = k_p
                assigned_hypersphere_p = hypersphere
                class_label_p = +1  # Positive class label
        
        for hypersphere in negative_hyperspheres:
            k_n = self.conformal_kernel(x, hypersphere.center, hypersphere, sigma)
            if k_n < min_value_n:
                min_value_n = k_n
                assigned_hypersphere_n = hypersphere
                class_label_n = -1  # Negative class label
    
        # Calculate contributions to center and boundary
        if min_value_p < min_value_n:  # Positive class
            #d_to_other_boundary = np.abs(np.linalg.norm(x - assigned_hypersphere_n.center) - assigned_hypersphere_n.radius)
            d_to_other_boundary = np.abs(min_value_n - assigned_hypersphere_n.radius)
            c_to_cen = 1 - 1 / np.sqrt(min_value_p**2 + gamma)
            c_to_boundary = 1 - 1 / (np.sqrt(d_to_other_boundary**2 + gamma))
            contribution = np.maximum(c_to_cen, c_to_boundary)
            
            assignm = (x, class_label_p, contribution)
            assignments.append(assignm)
            assigned_hypersphere_p.assignments.append(assignm)
            assigned_hypersphere_p.elements.append(x)
    
        elif min_value_p > min_value_n:  # Negative class
            #d_to_other_boundary = np.abs(np.linalg.norm(x - assigned_hypersphere_p.center) - assigned_hypersphere_p.radius)
            d_to_other_boundary = np.abs(min_value_p - assigned_hypersphere_p.radius)
            c_to_cen = 1 - 1 / np.sqrt(min_value_n**2 + gamma)
            c_to_boundary = 1 - 1 / (np.sqrt(d_to_other_boundary**2 + gamma))
            contribution = np.maximum(c_to_cen, c_to_boundary)
            
            assignm = (x, class_label_n, contribution)
            assignments.append(assignm)
            assigned_hypersphere_n.assignments.append(assignm)
            assigned_hypersphere_n.elements.append(x)
    
        else:
            assignments.append((x, 0, 1))  # Noise

    def fuzzy(self, data):
        assignments = []
        
        # Clear previous assignments
        for hs in self.positive_hyperspheres + self.negative_hyperspheres:
            hs.assignments = []
            hs.elements = []

        for x in data.values:
            self.fuzzy_contribution(x, self.positive_hyperspheres, self.negative_hyperspheres, assignments, self.c1, self.sigma)

            # Optimize hyperspheres
            for pos_hs, neg_hs in zip(self.positive_hyperspheres, self.negative_hyperspheres):
                self.optimize_hypersphere(pos_hs, self.negative_hyperspheres)
                self.optimize_hypersphere(neg_hs, self.positive_hyperspheres)

        return assignments

    def optimize_hypersphere(self, hypersphere, other_hyperspheres):
        initial_params = [hypersphere.radius] + list(hypersphere.center)
        result = minimize(self.objective, initial_params, args=(hypersphere, other_hyperspheres), constraints={'type': 'ineq', 'fun': lambda x: x[0]})  # R >= 0

        hypersphere.radius = result.x[0]
        hypersphere.center = result.x[1:]

    def objective(self, params, hypersphere, other_hyperspheres):
        radius = params[0]
        center = params[1:]

        pos_part = self.c1 * np.sum(s for _, _1, s in hypersphere.assignments)
        neg_part = np.sum(np.linalg.norm(x - center)**2 for hs in other_hyperspheres for x in hs.elements)
        return radius**2 + pos_part - neg_part
import numpy as np
import pandas as pd
from memory_profiler import memory_usage
import time
from wrappers import Hypersphere, predict as ccp_predict, optimize_hypersphere as cpp_optimize_hypersphere, fuzzy_contribution as cpp_fuzzy_contribution

class HyperionFuzzy:
    def __init__(self, num_clusters=2, gamma=1.0, sigma=0.005, E=1e-7, max_iterations=100):
        self.num_clusters = num_clusters
        self.gamma = gamma
        self.sigma = sigma
        self.E = E
        self.max_iterations = max_iterations
        self.positive_hyperspheres = []
        self.negative_hyperspheres = []

    def train(self, data, labels):
        start_time = time.time()
        mem_usage_before = memory_usage()[0]
        
        transformed_data = data.map(self.polynomial_mapping)

        self.positive_hyperspheres, self.negative_hyperspheres = self.initialize_hyperspheres(transformed_data, labels)

        for iteration in range(self.max_iterations):
            assignments = self.fuzzy(transformed_data)

            if all(len(hypersphere.assignments) == 0 for hypersphere in self.positive_hyperspheres) or \
               all(len(hypersphere.assignments) == 0 for hypersphere in self.negative_hyperspheres):
                break

        end_time = time.time()
        mem_usage_after = memory_usage()[0]

        print(f"Training Time: {end_time - start_time:.4f} seconds")
        print(f"Memory Usage: {mem_usage_after - mem_usage_before:.4f} MiB")

        return assignments

    def predict(self, new_data):
        transformed_data = new_data.map(self.polynomial_mapping)

        predictions = ccp_predict(transformed_data, self.positive_hyperspheres, self.negative_hyperspheres, self.sigma)
        return np.array(predictions)

    def initialize_hyperspheres(self, data, labels):
        class_p = data[labels == 1]
        class_n = data[labels == -1]
        
        if len(class_p) < self.num_clusters or len(class_n) < self.num_clusters:
            raise ValueError("Not enough data points to initialize clusters.")
        
        positive_hyperspheres = []
        negative_hyperspheres = []

        for _ in range(self.num_clusters):
            random_point_p = class_p.iloc[np.random.choice(class_p.shape[0])]
            random_point_n = class_n.iloc[np.random.choice(class_n.shape[0])]
            
            radius = np.linalg.norm(random_point_p - random_point_n) / 3
            
            positive_hyperspheres.append(
                Hypersphere(random_point_p.values, radius, class_p.values)
            )
            negative_hyperspheres.append(
                Hypersphere(random_point_n.values, radius, class_n.values)
            )

        return positive_hyperspheres, negative_hyperspheres

    def polynomial_mapping(self, x):
        return np.exp(x)

    def fuzzy(self, data):
        assignments = []
        
        # Reset assignments and elements for each hypersphere
        for hs in self.positive_hyperspheres + self.negative_hyperspheres:
            hs.assignments = []

        for x in data.values:
            cpp_fuzzy_contribution(
                x, 
                self.positive_hyperspheres,
                self.negative_hyperspheres,
                assignments,
                self.gamma,
                self.sigma,
                self.E
            )

            for pos_hs, neg_hs in zip(self.positive_hyperspheres, self.negative_hyperspheres):
                if pos_hs.assignments:
                    cpp_optimize_hypersphere(hypersphere=pos_hs,
                                             other_hyperspheres=self.negative_hyperspheres,
                                             c1=self.gamma)
                if neg_hs.assignments:
                    cpp_optimize_hypersphere(hypersphere=neg_hs,
                                             other_hyperspheres=self.positive_hyperspheres,
                                             c1=self.gamma)

        return assignments

import numpy as np
import pandas as pd
from memory_profiler import memory_usage
import time
from .wrappers import Hypersphere, predict as ccp_predict, optimize_hypersphere as cpp_optimize_hypersphere, fuzzy_contribution as cpp_fuzzy_contribution

class HyperionFuzzy:
    def __init__(self, num_clusters=2, gamma=1.0, sigma=0.005, E=1e-7, max_iterations=5):
        self.num_clusters = num_clusters
        self.gamma = gamma
        self.sigma = sigma
        self.E = E
        self.max_iterations = max_iterations
        self.positive_hyperspheres = []
        self.negative_hyperspheres = []

    def train(self, data: pd.DataFrame, labels: pd.Series):
        """Train the model using labeled data."""
        start_time = time.time()
        mem_usage_before = memory_usage()[0]

        # Fix: Use applymap() instead of map()
        transformed_data = data.applymap(self.polynomial_mapping)

        self.positive_hyperspheres, self.negative_hyperspheres = self.initialize_hyperspheres(transformed_data, labels)

        for iteration in range(self.max_iterations):
            assignments = self.fuzzy(transformed_data)

            # Stop training if no assignments are being made
            if all(len(hs.assignments) == 0 for hs in self.positive_hyperspheres) or \
               all(len(hs.assignments) == 0 for hs in self.negative_hyperspheres):
                break

        end_time = time.time()
        mem_usage_after = memory_usage()[0]

        print(f"Training Time: {end_time - start_time:.4f} seconds")
        print(f"Memory Usage: {mem_usage_after - mem_usage_before:.4f} MiB")

        return assignments

    def predict(self, new_data: pd.DataFrame):
        """Predict class labels for new data."""
        transformed_data = new_data.applymap(self.polynomial_mapping)
        predictions = ccp_predict(transformed_data, self.positive_hyperspheres, self.negative_hyperspheres, self.sigma)
        return np.array(predictions)

    def initialize_hyperspheres(self, data: pd.DataFrame, labels: pd.Series):
        """Initialize hyperspheres from labeled data."""
        class_p = data[labels == 1]
        class_n = data[labels == -1]
        
        if len(class_p) < self.num_clusters or len(class_n) < self.num_clusters:
            raise ValueError("Not enough data points to initialize clusters.")
        
        positive_hyperspheres = []
        negative_hyperspheres = []

        for _ in range(self.num_clusters):
            #  Fix: Convert Series to NumPy array
            random_point_p = class_p.iloc[np.random.choice(class_p.shape[0])].to_numpy()
            random_point_n = class_n.iloc[np.random.choice(class_n.shape[0])].to_numpy()
            radius = np.linalg.norm(random_point_p - random_point_n) / 3
            
            positive_hyperspheres.append(Hypersphere(random_point_p.tolist(), radius, class_p.to_numpy().tolist()))
            negative_hyperspheres.append(Hypersphere(random_point_n.tolist(), radius, class_n.to_numpy().tolist()))

        return positive_hyperspheres, negative_hyperspheres

    def polynomial_mapping(self, x):
        """Apply a polynomial transformation to the input."""
        return np.exp(x)

    def fuzzy(self, data: pd.DataFrame):
        """Perform the fuzzy contribution step and optimize hyperspheres."""
        assignments = []

        #  Reset assignments
        for hs in self.positive_hyperspheres + self.negative_hyperspheres:
            hs.assignments = []

        for x in data.to_numpy():
            #  Fix: Capture return values
            assigned_class, contribution = cpp_fuzzy_contribution(
                x, self.positive_hyperspheres, self.negative_hyperspheres, 
                self.gamma, self.sigma, self.E
            )

            #  Fix: Store assignments correctly
            if assigned_class == 1:
                self.positive_hyperspheres[0].assignments.append((x, assigned_class, contribution))
            elif assigned_class == -1:
                self.negative_hyperspheres[0].assignments.append((x, assigned_class, contribution))

            #  Fix: Add missing arguments for optimization
            for pos_hs, neg_hs in zip(self.positive_hyperspheres, self.negative_hyperspheres):
                if pos_hs.assignments:
                    cpp_optimize_hypersphere(pos_hs, self.negative_hyperspheres, self.gamma, 0.01, self.max_iterations, 1e-6)
                if neg_hs.assignments:
                    cpp_optimize_hypersphere(neg_hs, self.positive_hyperspheres, self.gamma, 0.01, self.max_iterations, 1e-6)

        return assignments
import pandas as pd
import numpy as np
from src.hyperion_fuzzy.main import HyperionFuzzy

# Generate synthetic data
np.random.seed(42)
data = pd.DataFrame(np.random.rand(100, 2), columns=["x1", "x2"])
labels = np.random.choice([1, -1], size=100)

# Initialize model
model = HyperionFuzzy(num_clusters=2, max_iterations=10)

# Train the model
assignments = model.train(data, labels)

# Predict using the trained model
new_data = pd.DataFrame(np.random.rand(10, 2), columns=["x1", "x2"])
predictions = model.predict(new_data)

print("Predictions:", predictions)
import time
import pandas as pd
import numpy as np
from memory_profiler import memory_usage
from src.hyperion_fuzzy.main import HyperionFuzzy

# Measure time and memory for training
start_time = time.time()
mem_usage_before = memory_usage()[0]

np.random.seed(42)
data = pd.DataFrame(np.random.rand(100, 2), columns=["x1", "x2"])
labels = np.random.choice([1, -1], size=100)

#Initialize model
model = HyperionFuzzy(num_clusters=2, max_iterations=10)

model.train(data, labels)

end_time = time.time()
mem_usage_after = memory_usage()[0]

print(f"Training Time: {end_time - start_time:.4f} seconds")
print(f"Memory Usage: {mem_usage_after - mem_usage_before:.4f} MiB")
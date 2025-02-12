import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.hyperion_fuzzy.HyperionFuzzy import HyperionFuzzy

# Load data from CSV file
file_path = "data/Student's Dropout and Academic Success.csv"
data = pd.read_csv(file_path)

# Assuming the CSV has columns 'x1' and 'x2' for features and 'label' for labels
# Adjust the column names based on your actual CSV file structure
features = data[['x1', 'x2']]
labels = data['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize model
model = HyperionFuzzy(num_clusters=2, max_iterations=10)

# Train the model
assignments = model.train(X_train, y_train)

# Predict using the trained model
predictions = model.predict(X_test)

print("Predictions:", predictions)

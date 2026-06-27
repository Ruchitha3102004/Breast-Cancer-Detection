import pickle
import numpy as np

# Load the model
model = pickle.load(open(r'C:\Users\chana\OneDrive\Desktop\mini project\brest cancer mini project\breast_cancer_detector.pickle', 'rb'))

# Example input (replace with actual feature values)
sample_input = np.array([[14.2, 20.5, 90.3, 600.2, 0.1, 0.15, 0.05, 0.03, 0.2, 0.06,
                          0.3, 1.5, 3.0, 40.0, 0.005, 0.01, 0.02, 0.01, 0.015, 0.001,
                          16.0, 25.0, 110.0, 800.0, 0.12, 0.2, 0.08, 0.05, 0.25, 0.08]])

# Make a prediction
prediction = model.predict(sample_input)

# Print the prediction
print('Prediction:', prediction)

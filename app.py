from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Define the path to the model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'breast_cancer_detector.pickle')

# Load the trained model once when the server starts
try:
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    model = None
    print(f"Model file not found at {MODEL_PATH}. Ensure the path is correct.")
except Exception as e:
    model = None
    print(f"An error occurred while loading the model: {e}")

# List of feature names in the correct order
FEATURE_NAMES = [
    'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness',
    'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry',
    'mean_fractal_dimension', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se',
    'symmetry_se', 'fractal_dimension_se', 'worst_radius', 'worst_texture',
    'worst_perimeter', 'worst_area', 'worst_smoothness', 'worst_compactness',
    'worst_concavity', 'worst_concave_points', 'worst_symmetry',
    'worst_fractal_dimension'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction="Model is not loaded. Contact the administrator.", probability=None)

    if request.method == 'POST':
        try:
            # Initialize a list to hold feature values
            features = []

            # Iterate through each feature to extract and validate data
            for feature in FEATURE_NAMES:
                value = request.form.get(feature)
                if value is None:
                    raise ValueError(f"Missing value for {feature}.")
                try:
                    # Convert the input to float
                    float_value = float(value)
                    features.append(float_value)
                except ValueError:
                    raise ValueError(f"Invalid input for {feature}: '{value}'. Must be a number.")

            # Convert features to a numpy array and reshape for prediction
            input_data = np.array([features])

            # Make prediction
            prediction = model.predict(input_data)[0]

            # Get prediction probabilities
            prediction_proba = model.predict_proba(input_data)[0]
            probability = max(prediction_proba) * 100  # Convert to percentage

            # Interpret the prediction
            result = 'Positive (Malignant)' if prediction == 1 else 'Negative (Benign)'

            return render_template('index.html', prediction=result, probability=f"{probability:.2f}%")

        except Exception as e:
            # Log the exception (you can expand this to use logging frameworks)
            print(f"Error during prediction: {e}")
            # Return a user-friendly error message without exposing internal details
            return render_template('index.html', prediction="An error occurred during prediction. Please check your inputs and try again.", probability=None)

    # If the request method isn't POST, simply render the home page
    return render_template('index.html')

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)

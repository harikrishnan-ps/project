from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained models
cnn_model = load_model('heart_disease_model.h5')  # CNN model
linear_model = joblib.load('linear_regression_model.pkl')  # Linear Regression model
kmeans = joblib.load('kmeans_model.pkl')  # K-means model

# Load scaler parameters
scaler_mean = np.load('scaler.npy')
scaler_scale = np.load('scaler_scale.npy')
scaler = StandardScaler()
scaler.mean_ = scaler_mean
scaler.scale_ = scaler_scale

# Simulated test dataset for Linear Regression graph (replace with actual data)
X_test = np.random.rand(20, 3)  # Example scaled features
y_test = np.random.rand(20)    # Example actual values

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        features = [float(request.form['age']), float(request.form['sex']), float(request.form['cp'])]
        features = np.array(features).reshape(1, -1)

        # Scale features
        features_scaled = scaler.transform(features)

        # Get selected model
        selected_model = request.form['model']

        if selected_model == 'cnn':
            # Reshape for CNN model
            features_scaled = features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1], 1)
            # Make prediction using CNN model
            prediction = cnn_model.predict(features_scaled)[0][0]  # Extract probability
            response = {
                "model": "cnn",
                "prediction": float(prediction),
                "status": "Heart Disease Detected" if prediction > 0.5 else "No Heart Disease Detected"
            }
            return jsonify(response)

        elif selected_model == 'linear_regression':
            # Make predictions using Linear Regression model
            predictions = linear_model.predict(X_test)
            graph_data = {
                "model": "linear_regression",
                "actual": y_test.tolist(),
                "predicted": predictions.tolist()
            }
            return jsonify(graph_data)

        elif selected_model == 'kmeans':
            # Predict cluster using K-means model
            cluster = kmeans.predict(features_scaled)[0]
            response = {
                "model": "kmeans",
                "cluster": int(cluster),
                "status": f"Cluster {cluster}: {'No Heart Disease' if cluster == 0 else 'Heart Disease Detected'}"
            }
            return jsonify(response)

        return jsonify({"error": "Invalid model selection."})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

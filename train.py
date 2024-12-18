import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import joblib

# Load dataset
df = pd.read_csv('Heart.csv')  # Replace with your actual CSV file path

# Preprocess data
df.rename(columns={'Heart Disease': 'target'}, inplace=True)
df['target'] = df['target'].map({'Presence': 1, 'Absence': 0})

# Select relevant features
features = ['Age', 'Sex', 'Chest pain type']
target = 'target'
X = df[features]
y = df[target]

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save scaler
np.save('scaler.npy', scaler.mean_)
np.save('scaler_scale.npy', scaler.scale_)

# 1. Train CNN Model
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

cnn_model = Sequential([
    Conv1D(32, kernel_size=2, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

cnn_history = cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=16, validation_data=(X_test_cnn, y_test))

cnn_model.save('heart_disease_model.h5')
print("CNN Model saved!")

# 2. Train Linear Regression Model
linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

joblib.dump(linear_model, 'linear_regression_model.pkl')
print("Linear Regression Model saved!")

# 3. Train K-means Model
kmeans = KMeans(n_clusters=3, random_state=42)

kmeans.fit(X)

joblib.dump(kmeans, 'kmeans_model.pkl')
print("K-means Model saved!")

# Evaluate models
cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test_cnn, y_test)
print(f"CNN Model accuracy: {cnn_accuracy * 100:.2f}%")

linear_score = linear_model.score(X_test, y_test)
print(f"Linear Regression Model R^2 score: {linear_score:.2f}")

# You may also want to print K-means cluster centers
print("K-means Cluster Centers:")
print(kmeans.cluster_centers_)

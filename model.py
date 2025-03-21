import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load the processed dataset
file_path = "processed_dataset.csv"
df = pd.read_csv(file_path)

# Define features and target
y = df["phone_duration"]  # Target (phoneme duration)
X = df.drop(columns=["phone_duration","phone","phone_class"])  # Features


# Split dataset into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor (Baseline Model)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Model RMSE: {rmse:.4f}")

# Save the trained model
joblib.dump(model, "phoneme_duration_model.pkl")
print("Model saved as phoneme_duration_model.pkl")

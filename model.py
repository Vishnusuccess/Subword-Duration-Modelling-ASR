import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm 

# Load the processed dataset
file_path = "processed_dataset.csv"
df = pd.read_csv(file_path)

# Filter dataset where phone_duration <= 0.3
df= df[df["phone_duration"] <= 0.3]


# Define features and target
y = df["phone_duration"] 
X = df.drop(columns=["phone_duration", "phone", "phone_class"])  

# Split dataset into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train model with progress tracking
for _ in tqdm(range(1), desc="Training the Random Forest Model", ncols=100):
    model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f"\nModel Performance Metrics:")
print(f"Model RMSE: {rmse:.4f}")
print(f"Model MAE: {mae:.4f}")
print(f"Model R²: {r2:.4f}")

# Save the trained model
joblib.dump(model, "phoneme_duration_model.pkl")
print("Model saved as phoneme_duration_model.pkl")

# ----- Visualizations -----

# 1. Predicted vs Actual Values (Scatter Plot)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label="Predictions")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2, label="Perfect Fit")
plt.title("Predicted vs Actual Phoneme Durations")
plt.xlabel("Actual Phoneme Duration")
plt.ylabel("Predicted Phoneme Duration")
plt.legend()
plt.grid(True)
plt.show()

# 2. Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.residplot(x=y_pred, y=residuals, lowess=True, color="green", line_kws={'color': 'red', 'lw': 2})
plt.axhline(y=0, color='black', linestyle='--', lw=2)  
plt.title("Residuals Plot")
plt.xlabel("Predicted Phoneme Duration")
plt.ylabel("Residuals (Actual - Predicted)")
plt.grid(True)
plt.show()

# 3. Feature Importance Plot
features = X.columns
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.barh(range(X.shape[1]), importances[indices], align="center", color="skyblue")
plt.yticks(range(X.shape[1]), [features[i] for i in indices])
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.title("Feature Importance (Random Forest)")
plt.gca().invert_yaxis()  
plt.show()

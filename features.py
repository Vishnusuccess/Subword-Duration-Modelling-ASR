import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
file_path = "extracted_data_all_folders.csv" 
df = pd.read_csv(file_path)

# Selected Features for modelling
selected_features = [
    "phone", "phone_class", "speech_energy", "noise_energy", "dynamic_ratio",
    "snr", "phone_start", "clipping", "phone_duration"  
]

# One-Hot Encode the phoneme class
encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
encoded_classes = encoder.fit_transform(df[['phone_class']])
class_columns = encoder.get_feature_names_out(['phone_class'])

df_encoded = pd.DataFrame(encoded_classes, columns=class_columns)
df = pd.concat([df, df_encoded], axis=1)

# Compute previous and next phoneme duration
df["prev_phone_duration"] = df["phone_duration"].shift(1).fillna(0)
df["next_phone_duration"] = df["phone_duration"].shift(-1).fillna(0)

# Compute speech rate (total sentence duration / number of phonemes)
df["speech_rate"] = df.groupby("sentence")["phone_duration"].transform(lambda x: x.sum() / len(x))

# Handle any potential NaN or infinite values
df = df.replace([np.inf, -np.inf], np.nan)  
df = df.fillna(0)  

# Check if there are any remaining infinite or NaN values
if df.isnull().values.any():
    print("Warning: There are still missing values in the dataset.")
else:
    print("No missing or infinite values found in the dataset.")

# Keep the necessary columns
final_features = selected_features + list(class_columns) + ["prev_phone_duration", "next_phone_duration", "speech_rate"]
df_final = df[final_features]

# Save the processed dataset
df_final.to_csv("processed_dataset.csv", index=False)
print("Preprocessing complete. Processed dataset saved as 'processed_dataset.csv'.")
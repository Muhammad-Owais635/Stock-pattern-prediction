#Liberaries to load model and dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib  # We'll use joblib to save and load the model


# Load and preprocess your dataset
data = pd.read_csv("Test.csv")
X = data
#Check these column if these columns are not in your dataset yo
X = data.drop("pattern", axis=1)  # Features

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Load the saved model for future use
model_filename = "stock_pattern_model_15m.pkl"
loaded_model = joblib.load(model_filename)

# Test the loaded model on the test set
y_test_pred = loaded_model.predict(X )

# Create a DataFrame for the predictions
predictions_df = pd.DataFrame({"Predicted": y_test_pred})

# Concatenate the original dataset (X_df) with the predictions
combined_df = pd.concat([data, predictions_df], axis=1)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv("combined_stock_data_15m.csv", index=False)

print("Predictions saved to combined_stock_data_15m.csv")

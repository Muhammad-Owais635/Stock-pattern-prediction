#Liberaries to train, save and load model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  # We'll use joblib to save and load the model

# Load and preprocess your dataset
data = pd.read_csv("stock_data_15m.csv")

X = data.drop("pattern", axis=1)  # Features
X = X.drop("interval", axis=1)  # Features
X = X.drop("index", axis=1)  # Features
y = data["pattern"]  # Labels


scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Create a random forest classifier model
model = RandomForestClassifier()

# Train the model on the training data
model.fit(X_train, y_train)

# Save the trained model to a file
model_filename = "stock_pattern_model_15m.pkl"
joblib.dump(model, model_filename)

# Test the loaded model on the test set
y_test_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", accuracy)

# Print the classification report
report = classification_report(y_test, y_test_pred)
print("Classification Report:\n", report)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestClassifier

# 1. Load the preprocessed data
input_file = "../Preprocessed Data/clinical_dataset_preprocessed.csv"
df = pd.read_csv(input_file)

# 2. Define the columns strictly forbidden by the project description
# These are the columns used to derive the 'fried' status.
forbidden_columns = [
    "weight_loss",
    "exhaustion_score",
    "gait_speed_slower",
    "grip_strength_abnormal",
    "low_physical_activity"
]

# 3. Separate Features (X) and Target (y)
# We drop the forbidden columns AND the target column 'fried' from X
X = df.drop(columns=["fried"] + forbidden_columns)
y = df["fried"]

# 4. Split the data into Training and Testing sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scale the features (Crucial for Neural Networks)
# NNs struggle if input features have vastly different ranges (e.g. Age vs BMI)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)





# 6. Initialize Random Forest (100 trees)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# 7. Train
rf_model.fit(X_train_scaled, y_train)

# 8. Predict
y_pred_rf = rf_model.predict(X_test_scaled)

# 9. Evaluate
print("\n--- Random Forest Results ---")
print(classification_report(y_test, y_pred_rf, target_names=['Non-frail', 'Pre-frail', 'Frail']))
print("Accuracy:", rf_model.score(X_test_scaled, y_test))
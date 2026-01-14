import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# 1. Load the preprocessed data
input_file = "../Preprocessed Data/clinical_dataset_preprocessed.csv"
df = pd.read_csv(input_file)

# 2. Define the columns strictly forbidden by the project description
forbidden_columns = [
    "weight_loss",
    "exhaustion_score",
    "gait_speed_slower",
    "grip_strength_abnormal",
    "low_physical_activity"
]

# 3. Separate Features (X) and Target (y)
X = df.drop(columns=["fried"] + forbidden_columns)
y = df["fried"]

# 4. Split the data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Initialize and Train Random Forest
# class_weight='balanced' helps with the imbalance between Frail/Non-frail classes
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)

# 7. Predict
y_pred_rf = rf_model.predict(X_test_scaled)

# 8. Evaluation Metrics
print("\n--- Random Forest Results ---")
target_names = ['Non-frail (0)', 'Pre-frail (1)', 'Frail (2)']
print(classification_report(y_test, y_pred_rf, target_names=target_names))
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred_rf)
print("\nConfusion Matrix (Numerical):")
print(cm)

# --- Visualization 1: Confusion Matrix Heatmap ---
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix: Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# --- Visualization 2: Feature Importance ---
# Get feature importances from the trained model
importances = rf_model.feature_importances_
feature_names = X.columns

# Create a DataFrame for visualization
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
# Sort by importance and take the top 10
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')
plt.title('Top 10 Clinical Features Predicting Frailty')
plt.xlabel('Importance Score')
plt.ylabel('Clinical Parameter')
plt.tight_layout()
plt.show()
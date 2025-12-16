import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

# 4. Split the data into Training and Testing sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Build the Neural Network
model = keras.Sequential([
    keras.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(3, activation='softmax')
])

# 7. Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 8. Train the model
print("Training Neural Network...")
history = model.fit(
    X_train_scaled, 
    y_train, 
    epochs=50,
    batch_size=8,
    validation_split=0.2,
    verbose=1
)

# 9. Evaluate on the Test Set
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest Set Accuracy: {accuracy*100:.2f}%")

# 10. Detailed Classification Report
y_pred_probs = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_probs, axis=1)

target_names = ['Non-frail (0)', 'Pre-frail (1)', 'Frail (2)']
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# --- Visualization 1: Confusion Matrix Heatmap ---
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (Numerical):")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix: Neural Network')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# --- Visualization 2: Training History (Accuracy & Loss) ---
# This shows if the model was learning or just memorizing
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 4))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()
import pandas as pd
import numpy as np
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

# 6. Build the Neural Network
model = keras.Sequential([
    # Input Layer
    keras.Input(shape=(X_train_scaled.shape[1],)),
    
    # Hidden Layer 1: 16 neurons, ReLU activation
    layers.Dense(16, activation='relu'),
    
    # Hidden Layer 2: 8 neurons, ReLU activation (optional, helps with complexity)
    layers.Dense(8, activation='relu'),
    
    # Output Layer: 3 neurons (Non-frail, Pre-frail, Frail)
    # We use 'softmax' to get a probability distribution across the 3 classes
    layers.Dense(3, activation='softmax')
])

# 7. Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', # Use sparse because y is integers (0,1,2), not one-hot encoded
    metrics=['accuracy']
)

# 8. Train the model
print("Training Neural Network...")
history = model.fit(
    X_train_scaled, 
    y_train, 
    epochs=50,          # Number of passes through the data
    batch_size=8,       # Update weights after every 8 samples
    validation_split=0.2, # Use 20% of training data to validate during training
    verbose=1
)

# 9. Evaluate on the Test Set
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest Set Accuracy: {accuracy*100:.2f}%")

# 10. Detailed Classification Report
y_pred_probs = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_probs, axis=1) # Convert probabilities to class labels (0,1,2)

target_names = ['Non-frail (0)', 'Pre-frail (1)', 'Frail (2)']
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# Optional: Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
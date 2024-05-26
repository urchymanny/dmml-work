import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv("fare_prediction_data/train.csv")


features = data[["trip_distance", "trip_time"]].values
labels = data['fare_amount'].values

# Scale the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

train_features, test_features, train_labels, test_labels = train_test_split(
    features,
    labels,
    test_size=0.2,
    random_state=42
)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_features, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels))


# Shuffle and batch the datasets
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32)
test_dataset = test_dataset.batch(32)

# Define the model with dropout layers for regularization
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(64, activation="relu",
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation="relu",
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=[
        tf.keras.metrics.MeanSquaredError(),
        tf.keras.metrics.MeanAbsoluteError(),
    ]
)

# Train the model
model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=20
)
model.save("regression.keras")

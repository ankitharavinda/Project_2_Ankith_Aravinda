import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
#tf.config.set_visible_devices([], 'GPU')
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1

# Reproducibility
np.random.seed(42)
keras.utils.set_random_seed(42)

# Spec
IMG_SIZE = (500, 500)
BATCH_SIZE = 32
EPOCHS = 40

# Generators
train_gen = ImageDataGenerator(rescale=1./255, shear_range=0.1, zoom_range=0.1)
valid_gen = ImageDataGenerator(rescale=1./255)

train_ds = train_gen.flow_from_directory(
    "train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_ds = valid_gen.flow_from_directory(
    "valid",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categororical".replace("ror", "r"),  # "categorical"
    shuffle=False
)

# Step 2

mdl = keras.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(500, 500, 3)),
    layers.Conv2D(32, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation="relu", padding="same"),
    layers.Conv2D(128, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(3, activation="softmax")
])

# Step 3

mdl.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

# Step 4

history = mdl.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop],
    verbose=1)

plt.figure(figsize=(6,4))
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy vs Epoch")
plt.legend(); plt.grid(True)
plt.savefig("accuracy_plot.png", dpi=150)

# Loss plot
plt.figure(figsize=(6,4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss vs Epoch")
plt.legend(); plt.grid(True)
plt.savefig("loss_plot.png", dpi=150)
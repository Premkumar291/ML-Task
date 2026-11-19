import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models


(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

print("Training images:", X_train.shape)
print("Training labels:", y_train.shape)
print("Testing images:", X_test.shape)
print("Testing labels:", y_test.shape)

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(filters=16,kernel_size=(3, 3),activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train,
    y_train,
    epochs=3,
    batch_size=32,
    validation_split=0.1
)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nTest loss:", round(test_loss, 4))
print("Test accuracy:", round(test_accuracy * 100, 2), "%")

index = 0

image = X_test[index]

predictions = model.predict(
    image[np.newaxis, ...],
    verbose=0
)
predicted_digit = np.argmax(predictions[0])

print("\nPredicted digit:", predicted_digit)
print("Actual digit:", y_test[index])

plt.imshow(image.squeeze(), cmap="gray")
plt.title(
    f"Predicted: {predicted_digit} | Actual: {y_test[index]}"
)
plt.axis("off")
plt.show()

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("CNN Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
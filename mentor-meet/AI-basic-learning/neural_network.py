from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

iris = load_iris()

X = iris.data
y = iris.target

print("Dataset loaded successfully!")
print("Input features:", iris.feature_names)
print("Flower species:", iris.target_names)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = keras.Sequential([
    layers.Input(shape=(4,)),             
    layers.Dense(10, activation="relu"),
    layers.Dense(3, activation="softmax")
])


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nNeural Network Architecture:")
model.summary()


print("\nTraining the neural network...")
training_model = model.fit(
    X_train,
    y_train,
    epochs=40,
    batch_size=8,
    verbose=2,
    validation_split=0.1
)


test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


predictions = model.predict(X_test, verbose=0)
predicted_classes = predictions.argmax(axis=1)

for i in range(5):
    predicted_species = iris.target_names[predicted_classes[i]]
    actual_species = iris.target_names[y_test[i]]
    print(f"\nFlower {i + 1}")
    print("Predicted species:", predicted_species)
    print("Actual species:   ", actual_species)


correct_predictions = (predicted_classes == y_test).sum()

print("\nCorrect predictions:",correct_predictions, "/", len(y_test))
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age", "Outcome"
]

df = pd.read_csv(url, names=columns)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Input(shape=(8,)),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()
model.fit(X_train, y_train, epochs=50, verbose=1)



loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test accuracy:", round(accuracy * 100, 2), "%")
print("Test loss:", round(float(loss), 2))

probability = model.predict(X_test[:1], verbose=0)[0][0]

print("Predicted probability:", round(float(probability), 2))

if probability >= 0.5:
    print("Predicted class: 1")
else:
    print("Predicted class: 0")

print("Actual class:", y_test.iloc[0])

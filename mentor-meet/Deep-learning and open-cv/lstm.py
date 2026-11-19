
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_excel("./dataset/Mobile_Reviews_Dataset.xlsx", engine="openpyxl")

print("Original dataset shape:", df.shape)
print(df.head())

df = df[["Review_Text", "Rating"]].copy()
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df = df.dropna(subset=["Review_Text", "Rating"])
df["Review_Text"] = df["Review_Text"].astype(str).str.strip()
df = df[df["Review_Text"] != ""]
df = df[df["Rating"].between(1, 5)]
df = df[df["Rating"] != 3]

df["Sentiment"] = (df["Rating"] >= 4).astype(int)

X = df["Review_Text"]
y = df["Sentiment"]

print("\nDataset shape:", df.shape)
print("\nSentiment distribution:")
print(y.value_counts().rename(index={0: "Negative", 1: "Positive"}))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
)

MAX_TOKENS = 10000
SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 32
LSTM_UNITS = 32
BATCH_SIZE = 32
EPOCHS = 2

vectorizer = layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH,
    standardize="lower_and_strip_punctuation"
)

vectorizer.adapt(tf.constant(X_train.tolist(), dtype=tf.string))

vocabulary_size = len(vectorizer.get_vocabulary())
print("\nVocabulary size:", vocabulary_size)

train_texts = tf.constant(X_train.tolist(), dtype=tf.string)
val_texts = tf.constant(X_val.tolist(), dtype=tf.string)
test_texts = tf.constant(X_test.tolist(), dtype=tf.string)

train_labels = tf.constant(y_train.to_numpy(), dtype=tf.float32)
val_labels = tf.constant(y_val.to_numpy(), dtype=tf.float32)
test_labels = tf.constant(y_test.to_numpy(), dtype=tf.float32)

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_texts, train_labels)
).shuffle(len(X_train), seed=42).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_texts, val_labels)
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_texts, test_labels)
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(classes, weights)
}

print("\nClass weights:", class_weights)

model = keras.Sequential([
    keras.Input(shape=(), dtype=tf.string),
    vectorizer,
    layers.Embedding(
        input_dim=MAX_TOKENS,
        output_dim=EMBEDDING_DIM,
        mask_zero=True
    ),
    layers.LSTM(LSTM_UNITS),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall")
    ]
)

model.summary()

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stopping]
)

print("\nTest results:")
test_results = model.evaluate(test_dataset, return_dict=True)

for metric, value in test_results.items():
    print(f"{metric}: {value:.4f}")

probabilities = model.predict(test_texts, verbose=0).flatten()
predictions = (probabilities >= 0.5).astype(int)

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Negative", "Positive"],
    zero_division=0
))

def predict_sentiment(review):
    probability = float(
        model.predict(
            tf.constant([review], dtype=tf.string),
            verbose=0
        )[0][0]
    )

    sentiment = "Positive" if probability >= 0.5 else "Negative"

    print("\nReview:", review)
    print("Predicted sentiment:", sentiment)

predict_sentiment("The battery life is excellent and the phone is fast")
predict_sentiment("The phone is slow and the camera is bad")
predict_sentiment("Good mobile at this price")

# model.save("mobile_review_lstm.keras")
print("\nModel saved as mobile_review_lstm.keras")

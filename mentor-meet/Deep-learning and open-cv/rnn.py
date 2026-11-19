import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# url = "https://raw.githubusercontent.com/yshrivastava592-creator/CodeAlpha_Flipkart_Mobile_Analysis/main/Mobile_Reviews_Dataset.xlsx"

df = pd.read_excel("./dataset/Mobile_Reviews_Dataset.xlsx", engine="openpyxl")

print(df.head())
print(df.columns.tolist())

df = df[["Review_Text", "Rating"]].copy()
df["Review_Text"] = df["Review_Text"].astype("string").str.strip()
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

df = df.dropna(subset=["Review_Text", "Rating"])
df = df[df["Review_Text"].str.len() > 0]
df = df[df["Rating"].between(1, 5)]
df = df[df["Rating"] != 3].copy()

df["sentiment"] = (df["Rating"] >= 4).astype("int32")

print("\nCleaned dataset shape:", df.shape)
print(df["sentiment"].value_counts())

X = df["Review_Text"].astype(str).to_numpy(dtype=str)
y = df["sentiment"].to_numpy(dtype=np.int32)

if len(np.unique(y)) < 2:
    raise ValueError("The dataset must contain both sentiment classes.")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training reviews:", len(X_train))
print("Testing reviews:", len(X_test))

MAX_WORDS = 10000
MAX_LENGTH = 100
EMBEDDING_DIM = 32
BATCH_SIZE = 32

vectorizer = layers.TextVectorization(
    max_tokens=MAX_WORDS,
    output_mode="int",
    output_sequence_length=MAX_LENGTH,
    standardize="lower_and_strip_punctuation"
)

vectorizer.adapt(
    tf.data.Dataset.from_tensor_slices(X_train).batch(64)
)

print("Vocabulary size:", len(vectorizer.get_vocabulary()))

model = tf.keras.Sequential([
    layers.Input(shape=(), dtype=tf.string),
    vectorizer,
    layers.Embedding(
        input_dim=MAX_WORDS,
        output_dim=EMBEDDING_DIM,
        mask_zero=True
    ),
    layers.SimpleRNN(32),
    layers.Dense(16, activation="relu"),
    
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        tf.constant(X_train.tolist(), dtype=tf.string),
        tf.constant(y_train, dtype=tf.int32)
    )
)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (
        tf.constant(X_test.tolist(), dtype=tf.string),
        tf.constant(y_test, dtype=tf.int32)
    )
)

train_dataset = train_dataset.shuffle(
    len(X_train), seed=42
)

validation_count = max(1, int(len(X_train) * 0.2))

validation_dataset = train_dataset.take(validation_count).batch(BATCH_SIZE)
train_dataset = train_dataset.skip(validation_count).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

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

print("Class weights:", class_weights)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=validation_dataset,
    callbacks=[early_stopping],
    class_weight=class_weights,
    verbose=1
)

test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)

print("Test loss:", round(float(test_loss), 4))
print("Test accuracy:", round(float(test_accuracy) * 100, 2), "%")

probabilities = model.predict(test_dataset, verbose=0).ravel()
predictions = (probabilities >= 0.5).astype("int32")

print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Negative", "Positive"],
    zero_division=0
))

def predict_review(review):
    review_tensor = tf.constant([str(review)], dtype=tf.string)
    probability = float(model.predict(review_tensor, verbose=0)[0][0])
    sentiment = "Positive" if probability >= 0.5 else "Negative"

    print("\nReview:", review)
    print("Predicted sentiment:", sentiment)

predict_review("The battery life is excellent and the phone is fast")
predict_review("The phone is fast and the camera looks good")
predict_review("Bad mobile at this price")

# model.save("mobile_review_rnn.keras")

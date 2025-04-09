# bash
#pip install tensorflow numpy matplotlib pandas tensorflow-datasets

#Load IMDb Movie Reviews Dataset
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Load IMDb dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)

# Pad sequences to ensure uniform input length
max_length = 200
x_train = pad_sequences(x_train, maxlen=max_length, padding='post')
x_test = pad_sequences(x_test, maxlen=max_length, padding='post')

print("Training data shape:", x_train.shape)
print("Test data shape:", x_test.shape)


#Define LSTM Model Architecture

from tensorflow.keras import layers, Sequential

def build_model(vocab_size=10000, embedding_dim=128, lstm_units=64):
    model = Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        layers.LSTM(lstm_units),
        layers.Dense(1, activation='sigmoid')  # Binary classification (positive/negative)
    ])
    return model

model = build_model()
model.summary()

#Specify Loss Function and Optimizer
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the Model
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_data=(x_test, y_test)
)

#Evaluate the Model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")


Inference
# Decode words from IMDb dataset index to text
word_index = tf.keras.datasets.imdb.get_word_index()

def decode_review(encoded_review):
    reverse_word_index = {value: key for key, value in word_index.items()}
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Example review (encoded format)
sample_review = x_test[0]
decoded_review = decode_review(sample_review)

# Predict sentiment
prediction = model.predict(sample_review.reshape(1, -1))
sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

print(f"Review: {decoded_review}")
print(f"Predicted Sentiment: {sentiment}")




#Visualize Training Results.

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

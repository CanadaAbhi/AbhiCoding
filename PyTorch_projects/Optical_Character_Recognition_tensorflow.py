#pip install tensorflow opencv-python matplotlib

import tensorflow as tf

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshape images to include channel dimension
x_train = x_train[..., tf.newaxis] / 255.0
x_test = x_test[..., tf.newaxis] / 255.0

# Convert labels to strings (for CTC loss compatibility)
y_train = [tf.strings.as_string(label) for label in y_train]
y_test = [tf.strings.as_string(label) for label in y_test]

from tensorflow.keras import layers, Model

def build_ocr_model(input_shape=(28, 28, 1), num_classes=11):  # 0-9 + blank
    inputs = layers.Input(shape=input_shape, name="image")
    
    # ResNet-inspired CNN backbone
    x = layers.Conv2D(32, (3,3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2,2))(x)
    
    x = layers.Conv2D(64, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2,2))(x)
    
    # Reshape for LSTM
    x = layers.Reshape((-1, 64))(x)
    
    # Bidirectional LSTM layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    return Model(inputs, outputs)


def ctc_loss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_pred)[0], dtype="int64")
    input_len = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    
    # Convert string labels to sparse tensor
    labels = tf.strings.unicode_decode(y_true, input_format="UTF-8")
    labels = tf.cast(labels, dtype="int32")
    
    # Calculate loss
    loss = tf.reduce_mean(tf.nn.ctc_loss(
        labels=labels,
        logits=y_pred,
        label_length=None,
        logit_length=tf.fill([batch_len], input_len),
        logits_time_major=False
    ))
    
    return loss

# Build and compile model
model = build_ocr_model()
model.compile(optimizer="adam", loss=ctc_loss, metrics=["accuracy"])

# Create TensorFlow Dataset
def create_dataset(images, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(x_train, y_train)
test_dataset = create_dataset(x_test, y_test)

# Train model
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=10,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3),
        tf.keras.callbacks.ModelCheckpoint("ocr_model.keras", save_best_only=True)
    ]
)
#Inference
def decode_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(
        pred, 
        input_length=input_len,
        greedy=True
    )[0][0]
    
    # Convert indices to characters
    output_text = tf.strings.reduce_join(
        tf.strings.as_string(results),
        axis=-1,
        separator=""
    )
    
    return output_text.numpy()

# Test on sample image
sample_image = x_test[0]
pred = model.predict(tf.expand_dims(sample_image, axis=0))
print("Predicted:", decode_predictions(pred)[0])
print("Actual:", y_test[0].numpy().decode())



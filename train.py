import tensorflow as tf
import tensorflow_datasets as tfds

# Load the Stanford Dogs Dataset
ds_train_full, ds_info = tfds.load(
    'stanford_dogs',
    split='train',
    as_supervised=True,
    with_info=True
)
ds_test = tfds.load(
    'stanford_dogs',
    split='test',
    as_supervised=True
)

# Determine the number of examples for splitting
train_size = tf.data.experimental.cardinality(ds_train_full).numpy()
val_size = int(0.2 * train_size) # Use 20% of the training data for validation
train_size = train_size - val_size

# Create training and validation datasets
train_ds = ds_train_full.take(train_size)
val_ds = ds_train_full.skip(train_size).take(val_size)

# Image size and number of classes
IMG_SIZE = 224
NUM_CLASSES = ds_info.features['label'].num_classes

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

# Prepare the datasets for training and validation
BATCH_SIZE = 32 # You might need to adjust this
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(preprocess).shuffle(buffer_size=train_size).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(preprocess).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

# Load the base MobileNetV2 model (without the top)
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False # Freeze the base model

# Create the classification head
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')

# Build the final model
model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
EPOCHS = 10 # Start with a small number of epochs
history = model.fit(train_ds,
                    epochs=EPOCHS,
                    validation_data=val_ds)

# Save the trained model
model.save('dog_breed_model.h5')

print("Training complete. Model saved as dog_breed_model.h5")
import os  # Import operating system module
import PIL  # Import Python Imaging Library
import shutil  # Import file operation utilities
import pathlib  # Import object-oriented filesystem paths
import numpy as np  # Import numerical computing library
import matplotlib.pyplot as plt  # Import plotting library

import tensorflow as tf  # Import TensorFlow library
from tensorflow import keras  # Import Keras from TensorFlow
from tensorflow.keras import layers  # Import layers from Keras
from tensorflow.keras.models import Sequential, save_model  # Import Sequential model and save_model function


dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"  # Define URL for dataset
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)  # Download and extract dataset
data_dir = pathlib.Path(data_dir)  # Convert directory path to pathlib object


image_count = len(list(data_dir.glob('*/*.jpg')))  # Count total number of images in dataset
print("Total no. of images: ", image_count)  # Print total number of images


batch_size = 32  # Define batch size for training
img_height = 180  # Define image height
img_width = 180  # Define image width


train_ds = tf.keras.preprocessing.image_dataset_from_directory(  # Create training dataset
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(  # Create validation dataset
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)


class_names = train_ds.class_names  # Get class names from training dataset
print("Class Names :", class_names)  # Print class names


train_ds = train_ds.cache().shuffle(1000)  # Cache and shuffle training dataset
val_ds = val_ds.cache()  # Cache validation dataset


num_classes = len(class_names)  # Determine number of classes


model = Sequential([  # Define model architecture using Sequential API
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])


model.compile(optimizer='adam',  # Compile model with optimizer, loss function, and metrics
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


print(model.summary())  # Display model summary


print("Model Training....")  # Print training start message
epochs=10  # Define number of epochs for training
history = model.fit(  # Train the model
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
print("Training Complete!")  # Print training completion message


acc = history.history['accuracy']  # Retrieve training accuracy
val_acc = history.history['val_accuracy']  # Retrieve validation accuracy

loss = history.history['loss']  # Retrieve training loss
val_loss = history.history['val_loss']  # Retrieve validation loss

epochs_range = range(epochs)  # Define range of epochs


plt.figure(figsize=(8, 8))  # Create a figure for plotting
plt.subplot(1, 2, 1)  # Define subplot for training/validation accuracy
plt.plot(epochs_range, acc, label='Training Accuracy')  # Plot training accuracy
plt.plot(epochs_range, val_acc, label='Validation Accuracy')  # Plot validation accuracy
plt.legend(loc='lower right')  # Add legend to the plot
plt.title('Training and Validation Accuracy')  # Add title to the plot


plt.subplot(1, 2, 2)  # Define subplot for training/validation loss
plt.plot(epochs_range, loss, label='Training Loss')  # Plot training loss
plt.plot(epochs_range, val_loss, label='Validation Loss')  # Plot validation loss
plt.legend(loc='upper right')  # Add legend to the plot
plt.title('Training and Validation Loss')  # Add title to the plot
plt.show()  # Display the plot


save_model(model, 'flower_model_trained.hdf5')  # Save the trained model to a file
print("Model Saved")  # Print model saved message
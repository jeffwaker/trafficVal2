#!/usr/bin/env python3

# ### Import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import glob
import matplotlib.pyplot as plt


# ### Verify dataset directory and contents

# The dataset is a subset of the GTSRB from kaggle
# The data directory shoud contain both the training and the test set
dataDirectory = ('C:/Users/Jeff/Desktop/GTSDS/')
trainingDirectory = dataDirectory+'training/'
testDirectory = dataDirectory+'test/'
print (dataDirectory)
print (trainingDirectory)
print (testDirectory)


# find the number of all the recursive files listed in the training directory
# The total number of the GTSRB should be 23519 files
files = glob.glob(trainingDirectory + '*/*.png')
imageCount = len(files)
print(imageCount, "files found in" , trainingDirectory)


# ###  build training and val set

# Parameter defines to load.
# resized image to 50 to keep consistancy and speed up training.
# Smaller image sizes causes accuracy to decreases drastically
batchSize = 32
resizeHeight = 50
resizeWidth = 50


# Split the data into 80% train and 20% validation
trainDataset = tf.keras.preprocessing.image_dataset_from_directory(
  trainingDirectory,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(resizeHeight, resizeWidth),
  batch_size=batchSize)

validationDataset = tf.keras.preprocessing.image_dataset_from_directory(
  trainingDirectory,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(resizeHeight, resizeWidth),
  batch_size=batchSize)


# List the classes
class_names = validationDataset.class_names
print("List of classes")
print(class_names)


# ### Visualize the data

# Check that the data matches with the classes they belong to before training
plt.figure(figsize=(10, 10))
for images, labels in trainDataset.take(1):
  for i in range(12):
    ax = plt.subplot(4, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")


# Create datasets for model to train on.
# The batch for the images should be shaped as (32x50x50x3)
for imageBatch, labelsBatch in trainDataset:
  image_batch_np = np.stack(list(imageBatch))
  labels_batch_np = np.stack(list(labelsBatch))
  break

print(image_batch_np.shape)
print(labels_batch_np.shape)


# ### prepare dataset for optimal training

# shuffle and store the training and validation images in cache to try to prevent bottlenecks while during training
AUTOTUNE = tf.data.AUTOTUNE
trainDataset = trainDataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validationDataset = validationDataset.cache().prefetch(buffer_size=AUTOTUNE)


# ### Normalization Layer

# The last column of the image batch are the color channel values, represented in range 0x00 - 0xFF. Rescale the values to be between zero and 1 for the normaization layer
# note(this is a exerimental function in keras and is considered volatile)
normalizationLayer = layers.experimental.preprocessing.Rescaling(1./255)

# Apply rescaled layer to the dataset
normalizedDataset = trainDataset.map(lambda x, y: (normalizationLayer(x), y))
image_batch_np, labels_batch_np = next(iter(normalizedDataset))
first_image = image_batch_np[0]

print(np.min(first_image), np.max(first_image)) 


# ### Data augmentation

# Augment the training data to reduce overfitting
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomContrast(0.05, input_shape=(resizeHeight, resizeWidth, 3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ])

#plot samples of the augmented samples
plt.figure(figsize=(10, 10))
for images, _ in trainDataset.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))


# ### Dropout

# add the dropout layer and data augmentation layers tro the model
numberClasses = 23
model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(numberClasses)
])


# ### Compile and train the model
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.summary()

epochs = 15
history = model.fit(trainDataset, validation_data=validationDataset, epochs=epochs)


# ## Visualize training results

# After applying data augmentation and Dropout, there is less overfitting than before, and training and validation accuracy are closer aligned. 
accuracy = history.history['accuracy']
validationAccuracy = history.history['val_accuracy']
loss = history.history['loss']
validLoss = history.history['val_loss']
epochsRange = range(epochs)

# set up the figures to be able to plot the results of the model
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochsRange, accuracy, label='Training Accuracy')
plt.plot(epochsRange, validationAccuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('validation accuracy with training accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochsRange, loss, label='Training Loss')
plt.plot(epochsRange, validLoss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('validation Loss with training loss')
plt.show()


# ### Prediction

# Use the model to predict a ramdomly selected test image
testlist = list(glob.glob(testDirectory+ '*.png'))
randimage = random.randint(0, (len(testlist)))

randimage =str(randimage)
randimage =(randimage.zfill(5))

testpath = (testDirectory + str(randimage) + '.png')
print(testpath)

# Create a batch
img = keras.preprocessing.image.load_img(testpath, target_size=(resizeHeight, resizeWidth))
imageArray = keras.preprocessing.image.img_to_array(img)
imageArray = tf.expand_dims(imageArray, 0) 

predictions = model.predict(imageArray)
score = tf.nn.softmax(predictions[0])

print(100 * np.max(score), "%")
print(format(class_names[np.argmax(score)]))

Image.open(str(testpath))






# we have an image data in many folders and many formats
# we need to convert them to a single format which is "jpg" and put them in the same folder in new directory

import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# path to the directory where the images are
path = 'RGB arSL dataset'
# path to the directory where the images will be copied
path2 = 'sobel_edges'

# # create the directory
# os.mkdir(path2)

# # loop over the folders in the directory
# for folder in os.listdir(path):
#     print('folder :', folder)
#     # create the directory in the new directory
#     os.mkdir(os.path.join(path2, folder))
#     # loop over the images in the folder
#     for image in tqdm(os.listdir(os.path.join(path, folder))):
#         # read the image
#         img = cv2.imread(os.path.join(path, folder, image))
#         # convert the image to jpg format
#         cv2.imwrite(os.path.join(path2, folder, image[:-4] + '.jpg'), img)

import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from keras.models import Model, load_model
#from keras.layers import Dense, Input, Conv2D, MaxPool2D, Flatten,
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPool2D, Flatten ,Activation,Dropout
import glob
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
images=tf.keras.utils.image_dataset_from_directory(
    path2,
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    validation_split=0.2,
    seed=123,
    subset='training',
    interpolation='bilinear')
images_validation=tf.keras.utils.image_dataset_from_directory(
    path2,
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    validation_split=0.2,
    seed=123,
    subset='validation',
    interpolation='bilinear')
class_names = images.class_names
#print(class_names)
#print(len(class_names))
plt.figure(figsize=(10, 10))
for image, label in images.take(1):
  for i in range(25):
    ax = plt.subplot(5, 5, i +1)
    plt.imshow(image[i].numpy().astype("uint8"))
    plt.title(class_names[label[i]])
    plt.axis("off")
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
import tensorflow.keras as keras

resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(256,256,3),pooling='max')

output = resnet.layers[-1].output
output = tf.keras.layers.Flatten()(output)
resnet = Model(resnet.input, output)

res_name = []
for layer in resnet.layers:
    res_name.append(layer.name)
resnet.trainable=False
set_trainable = False
for layer in resnet.layers:
     if layer.name in res_name[-22:]:
         set_trainable = True
     if set_trainable:
         layer.trainable = True
     else:
         layer.trainable = False
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout

model6 = Sequential()
model6.add(resnet),
Dropout(0.5),
model6.add(Dense(31, activation='softmax'))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) 
model6.compile(optimizer = optimizer, loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
EPOCHS = 10
history6 = model6.fit(
    images,
    epochs = EPOCHS,
    batch_size = 128,
    validation_data=images_validation,callbacks=[early_stop]
) 
# Plotting training & validation accuracy values
plt.figure(figsize=(10, 5))
plt.plot(history6.history['accuracy'], label='Train')
plt.plot(history6.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Plotting training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history6.history['loss'], label='Train')
plt.plot(history6.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

# Function to prepare and make predictions on the validation set
def evaluate_model(model, dataset):
    val_predictions = []
    val_true = []
    for batch in dataset:
        imgs, labels = batch
        preds = model.predict(imgs)
        preds = np.argmax(preds, axis=1)  # Convert probabilities to class indices
        val_predictions.extend(preds)
        val_true.extend(labels.numpy())

    # Generate classification report
    cm = confusion_matrix(val_true, val_predictions)
    plt.figure(figsize=(12, 12))  # Increase figure size
    sns.set(font_scale=0.7)
    sns.heatmap(cm, annot=True, fmt="d", cmap='Reds', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()
    accuracy = accuracy_score(val_true, val_predictions)
    print(f'Accuracy: {accuracy}')
    return classification_report(val_true, val_predictions, target_names=class_names)

# Evaluate the model
report = evaluate_model(model6, images_validation)
print(report)

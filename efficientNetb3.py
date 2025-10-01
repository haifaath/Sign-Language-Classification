import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Setup Matplotlib preferences
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

# Path to the directory where the images are stored and will be loaded from
path2 = 'grayscale_images'  # Update this path as needed

# Load image data from directories
images = tf.keras.utils.image_dataset_from_directory(
    path2,
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    validation_split=0.2,
    seed=123,
    subset='training',
    label_mode='int',
    interpolation='bilinear'
)

images_validation = tf.keras.utils.image_dataset_from_directory(
    path2,
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    validation_split=0.2,
    seed=123,
    subset='validation',
    label_mode='int',
    interpolation='bilinear'
)


early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
efficientnet = EfficientNetB3(include_top=False, weights='imagenet', input_shape=(256, 256, 3), pooling='max')

for layer in efficientnet.layers:
    layer.trainable = False

model = Sequential([
    efficientnet,
    Dense(512, activation='relu'),
    #Dropout(0.5),
    Dense(len(images.class_names), activation='softmax')
])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01) 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    images,
    epochs=40,
    validation_data=images_validation,
    callbacks=[early_stop]
)

# Plotting training & validation accuracy values
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Plotting training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Evaluate the model
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
    plt.figure(figsize=(12, 12))
    sns.set(font_scale=1)
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=images.class_names, yticklabels=images.class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()
    accuracy = accuracy_score(val_true, val_predictions)
    print(f'Accuracy: {accuracy}')
    print(classification_report(val_true, val_predictions, target_names=images.class_names))

evaluate_model(model, images_validation)

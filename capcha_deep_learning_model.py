import os
os.chdir(r'C:\Users\Akash\PycharmProjects\Translation_NLP')
import cv2
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_directory = r'D:\python\datasets\capcha_raj_gov\Structured\train'
valid_directory = r'D:\python\datasets\capcha_raj_gov\Structured\valid'
test_directory =r'D:\python\datasets\capcha_raj_gov\Structured\test'

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_directory,
        target_size=(40, 40),
        color_mode='grayscale',
        shuffle=True,
        batch_size=32,
        class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
        valid_directory,
        target_size=(40, 40),
        color_mode='grayscale',
        shuffle=True,
        batch_size=32,
        class_mode='categorical')

model = tf.keras.Sequential([
tf.keras.layers.Conv2D(filters=32,kernel_size=3, padding="same", activation="relu", input_shape=[40, 40, 1]),
tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'),
tf.keras.layers.Conv2D(filters=32,kernel_size=3, padding="same", activation="relu"),
tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(units=128, activation='relu'),
tf.keras.layers.Dense(units=10, activation='softmax')
])
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
batch_size = 32
X_train_samples = 385
X_test_samples = 124
steps_per_epoch = int( np.ceil(X_train_samples / batch_size) )
validation_steps = int( np.ceil(X_test_samples / batch_size) )
history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=validation_steps)
model.save('Capcha_model')
model = tf.keras.models.load_model('Capcha_model')
test_img = cv2.imread(r'D:\python\datasets\capcha_raj_gov\bfc9c24a-00f9-11eb-809b-7cd30a81cb9a.png')
test_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
test_img.shape
test_img = test_img.reshape(1,40,40,1)
test_img = np.float32(test_img)
print(np.argmax(model.predict(test_img)))
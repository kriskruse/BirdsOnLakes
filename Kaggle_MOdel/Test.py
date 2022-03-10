import numpy as np

np.random.seed(11)
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import os
from keras import layers, optimizers, losses, metrics, callbacks, initializers
import matplotlib.pyplot as plt
import keras
from keras.applications.efficientnet_v2 import EfficientNetV2B3
from IPython.display import clear_output

plt.style.use("ggplot")
# %matplotlib inline
# import cv2
from PIL import Image


class PlotLearning(keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """

    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        # Plotting
        metrics = [x for x in logs if 'val' not in x]

        f, axs = plt.subplots(1, len(metrics), figsize=(15, 5))
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2),
                        self.metrics[metric],
                        label=metric)
            if logs['val_' + metric]:
                axs[i].plot(range(1, epoch + 2),
                            self.metrics['val_' + metric],
                            label='val_' + metric)

            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.show()


TRAIN = 'train'
VALID = 'valid'
TEST = 'test'

IMAGE_SIZE = (224, 224)

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN,
    label_mode='categorical',
    image_size=IMAGE_SIZE
)
class_names = train_data.class_names

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    VALID,
    label_mode='categorical',
    image_size=IMAGE_SIZE,

)
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    TEST,
    label_mode='categorical',
    image_size=IMAGE_SIZE,
    shuffle=False
)

train_data_pf = train_data.prefetch(buffer_size=tf.data.AUTOTUNE)
val_data_pf = val_data.prefetch(buffer_size=tf.data.AUTOTUNE)
test_data_pf = test_data.prefetch(buffer_size=tf.data.AUTOTUNE)

inputs = layers.Input(shape=(224, 224, 3), name='input_layer')
base_model = EfficientNetV2B3(include_top=False)
base_model.trainable = False
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D(name='Global_Average_Pool_2D')(x)
num_classes = len(train_data.class_names)
outputs = layers.Dense(num_classes, activation='softmax', dtype=tf.float32, name="Output_layer")(x)
model = keras.Model(inputs, outputs, name="model")

model.summary()
model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy']
)

EPOCHS = 25
history_of_model = model.fit(
    train_data_pf,
    epochs=EPOCHS,
    steps_per_epoch=int((1 / EPOCHS) * len(train_data_pf)),
    validation_data=val_data_pf,
    validation_steps=len(val_data_pf),
    callbacks=[PlotLearning()]
)

model_0_result = model.evaluate(test_data_pf)
print(model_0_result)

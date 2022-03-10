import keras.optimizers
import numpy as np
import os
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from keras.applications import efficientnet
from keras.applications import efficientnet_v2
from keras import layers, optimizers, losses, metrics, callbacks, initializers
from keras import Sequential, Model, Input
from IPython.display import clear_output
from keras.applications.efficientnet_v2 import EfficientNetV2B3

plt.style.use("ggplot")


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


def scheduler(epoch, lr):
    """
    Is a function that the model uses to get a better learning rate

    :param epoch: (int) Current epoch of model
    :param lr: (float) Current learning rate of model
    :return: (float) new learning rate to use
    """
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def load_data(path, img_h, img_w, shuf=True):
    """
    :param path: (str) Path of train data
    :param img_h: (int) Height of images
    :param img_w: (int) Width of images
    :param shuf: (Bool) Whether or not to shuffle the data
    :return dataset: Train dataset
    """
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=(img_h, img_w),
        label_mode='categorical',
        shuffle=shuf)

    return dataset


if __name__ == "__main__":
    # params for tuning
    EPOCHS = 25  # how many iterations to train
    batch = 25  # size of batches (This is dependent on GPU/CPU mem size, higher is faster but more mem is needed)
    seed = 42069

    # seeding
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # import the data as a dataset
    print("Train dataset: ")
    train_data = load_data("train", 224, 224)
    print("Test dataset: ")
    test_data = load_data("test", 224, 224, shuf=False)
    print("Validation dataset: ")
    val_data = load_data("valid", 224, 224)

    # we prefetch the data from cache
    train_data_pf = train_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_data_pf = test_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_data_pf = val_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # setup the model
    inputs = layers.Input(shape=(224, 224, 3), name='input_layer')
    base_model = EfficientNetV2B3(include_top=False)
    base_model.trainable = False
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name='Global_Average_Pool_2D')(x)
    num_classes = len(train_data.class_names)
    outputs = layers.Dense(num_classes, activation='softmax', dtype=tf.float32, name="Output_layer")(x)
    model = keras.Model(inputs, outputs, name="model")

    # Print out the structure of the model
    model.summary()
    # Compile model so we can use it
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer='adam',
        metrics=['accuracy']
    )

    history_of_model = model.fit(
        train_data_pf,
        epochs=EPOCHS,
        steps_per_epoch=int((1 / EPOCHS) * len(train_data_pf)),
        validation_data=val_data_pf,
        validation_steps=len(val_data_pf),
        callbacks=[PlotLearning()]
    )

    model_result = model.evaluate(test_data_pf)
    print("Evaluated on unseen test-data")
    print(f"""Loss on test: {model_result[0]}
    Accuracy on test {model_result[1]}""")

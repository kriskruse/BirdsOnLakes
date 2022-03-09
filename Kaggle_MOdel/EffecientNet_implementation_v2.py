import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications import efficientnet


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


def load_data(path, img_h, img_w, batch_size=10):
    """
    :param path: (str) Path of train data
    :param img_h: (int) Height of images
    :param img_w: (int) Width of images
    :param batch_size: (int) size of batches
    :return dataset: Train dataset
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=(img_h, img_w),
        batch_size=batch_size)

    return dataset


if __name__ == "__main__":
    # params for tuning
    epochs = 1      # how many iterations to train

    # import the data as a dataset
    print("Train dataset: ")
    train_ds = load_data("train", 224, 224)
    print("Test dataset: ")
    test_ds = load_data("test", 224, 224)
    print("Validation dataset: ")
    val_ds = load_data("valid", 224, 224)

    # setup the model
    model = efficientnet.EfficientNetB0(
        include_top=True,
        weights='imagenet')
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    model.compile(tf.keras.optimizers.SGD(), loss='mse')
    print("Model assigned successfully")

    # we prefetch the data from cache
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # fit the model
    model_fit = model.fit(train_ds, epochs=epochs, callbacks=[callback])
    model.summary()
    evaluation = model.evaluate(test_ds, callbacks=[callback])
    print(evaluation)

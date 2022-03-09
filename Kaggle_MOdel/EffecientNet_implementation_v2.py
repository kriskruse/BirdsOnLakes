import keras.optimizers
import numpy as np
import os
import tensorflow as tf
import tensorflow_addons as tfa
from keras.applications import efficientnet
from keras.applications import efficientnet_v2
from keras import layers, optimizers, losses, metrics, callbacks, initializers
from keras import Sequential, Model, Input


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
    epochs = 25  # how many iterations to train
    batch = 25  # size of batches (This is dependent on GPU/CPU mem size, higher is faster but more mem is needed)

    # import the data as a dataset
    print("Train dataset: ")
    train_ds = load_data("train", 224, 224, batch_size=batch)
    print("Test dataset: ")
    test_ds = load_data("test", 224, 224, batch_size=batch)
    print("Validation dataset: ")
    val_ds = load_data("valid", 224, 224, batch_size=batch)

    # setup the model
    # model = efficientnet_v2.EfficientNetV2B0(
    #     include_top=True, weights=None, classes=400)
    model = efficientnet_v2.EfficientNetV2B3(include_top=False, input_shape=(224, 224, 3), weights='imagenet')

    model.trainable = False
    # create and add input layer to model
    inputs = Input(shape=(224, 224, 3))
    main = model(inputs, training=False)

    # add more layers to our model
    x = layers.GlobalMaxPooling2D()(main)
    x = layers.Dense(256, 'relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(400, activation='softmax', name='outputs')(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    callback = callbacks.LearningRateScheduler(scheduler)
    reducer = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        patience=1,
        verbose=1,
        factor=0.1)

    # model.compile(tf.keras.optimizers.SGD(), loss='mse', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    # model.compile(optimizer='adam',
    #               metrics=[metrics.CategoricalAccuracy(name='accuracy'), tfa.metrics.F1Score(350),
    #                        metrics.TopKCategoricalAccuracy(k=5)],
    #               loss=losses.CategoricalCrossentropy(label_smoothing=0.1))

    model.compile(optimizer='adam', loss=losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
    model.summary()
    print("Model assigned successfully")

    # we prefetch the data from cache
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # fit the model
    #model_fit = model.fit(train_ds, epochs=epochs, callbacks=[callback])
    #model_fit = model.fit(train_ds, epochs=epochs)
    model_fit = model.fit(train_ds, callbacks=[callback], epochs=epochs)


    # eval the model and show the metrics
    model_eval = model.evaluate(test_ds, callbacks=[callback], return_dict=True)
    print(model_eval)

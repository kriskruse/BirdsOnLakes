import numpy as np
import os
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import efficientnet

# Force CPU use
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# data loader
def loadData(csv_path):
    """
    :param csv_path: Takes the path of a csv file containing image path and class of image
    :return: df_csv, class_list, paths, labels
    """

    df_csv = pd.read_csv(csv_path)
    class_list = df_csv["class index"]
    paths = df_csv["filepaths"]
    labels = df_csv["labels"]
    data_set = df_csv["data set"]
    return df_csv, class_list, paths, labels, data_set


def image_to_numpyarray(paths):
    """
    :param paths:
    :return: list of numpy arrays, of the images
    """
    image_list = []

    for path in paths:
        image = Image.open(path)
        image_list.append(np.asarray(image))
        image.close()
    return np.stack(image_list)


# Use a custom callback function to navigate different Learning rates for the fit function
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


if __name__ == "__main__":
    df_csv, _, _, _, _ = loadData("birds.csv")
    df_train = df_csv[df_csv.values == "train"]
    df_test = df_csv[df_csv.values == "test"]
    df_val = df_csv[df_csv.values == "valid"]

    paths_train = df_train["filepaths"]
    paths_test = df_test["filepaths"]
    paths_val = df_val["filepaths"]
    print("Loaded data successfully")

    model = efficientnet.EfficientNetB0(include_top=True,
                                        weights='imagenet',
                                        )
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    model.compile(tf.keras.optimizers.SGD(), loss='mse')
    print("Model assigned successfully")

    # TODO: make a function that saves the data so we dont have to convert all the image data everytime
    X_train = image_to_numpyarray(paths_train)
    Y_train = np.array(df_train["class index"])
    X_test = image_to_numpyarray(paths_test)
    Y_test = np.array(df_test["class index"])
    X_val = image_to_numpyarray(paths_val)
    Y_val = np.array(df_val["class index"])
    print("Image data converted successfully")

    # X_train = tf.data.Dataset.batch(X_train, batch_size=100)
    # Y_train = tf.data.Dataset.batch(Y_train, batch_size=100)
    # X_test = tf.data.Dataset.batch(X_test, batch_size=100)
    # Y_test = tf.data.Dataset.batch(Y_test, batch_size=100)
    # X_val = tf.data.Dataset.batch(X_val, batch_size=100)
    # Y_val = tf.data.Dataset.batch(Y_val, batch_size=100)
    # print("Batching succesful")

    # TODO: figure out why the model doesn't want to fit
    model_fit = model.fit(X_train, Y_train, epochs=1, callbacks=[callback])
    model_fit.summary()




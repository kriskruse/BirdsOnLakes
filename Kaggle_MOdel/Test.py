import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf

batch_size = 10
img_height = 224
img_width = 224

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


train_ds = tf.keras.utils.image_dataset_from_directory(
      "train",
      image_size=(img_height, img_width),
      batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
      "test",
      image_size=(img_height, img_width),
      batch_size=batch_size)


model = tf.keras.applications.efficientnet.EfficientNetB0(
                                include_top=True,
                                weights='imagenet'
                                )
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
model.compile(tf.keras.optimizers.SGD(), loss='mse')
print("Model assigned successfully")

normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))




AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

model_fit = model.fit(train_ds, validation_data=test_ds, epochs=25, callbacks=[callback])
model_fit.summary()

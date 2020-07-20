
import numpy as np

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential ,Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU



class pixelizedIoU(keras.losses.Loss):

    def __init__(self, degree=3, cyclic=True, reduction=keras.losses.Reduction.AUTO,
                 name="pixelized_iou"):
        super().__init__(reduction=reduction, name=name)
        self.degree = degree
        self.cyclic = cyclic

    def call(self, y_true, y_pred):

        batch_size = tf.shape(y_true)[0]

        num_knots = int(y_pred.shape[1]/2)

        cyclical = self.cyclic
        degree = self.degree

        max_pos = num_knots if cyclical else num_knots - degree
        knots = y_pred

        positions = tf.expand_dims(
                tf.range(start=0.0, limit=max_pos, delta=0.01, dtype=knots.dtype), axis=-1)

        tf.print(knots)

        spline = bspline.interpolate(knots, positions, degree, cyclical)
        spline = tf.squeeze(spline, axis=1)

        y_pred = tf.Variable(tf.zeros((batch_size, 256, 256, 1)))

        for i in batch_size:
            for step in spline:
                col = int(step[0])
                row = int(step[1])
                if y_pred[i, row, col, 0] == 0.0:
                    y_pred[i, row, col, 0].assign(1.0)

        ce = tf.losses.binary_crossentropy(
            y_true, y_pred, from_logits=True)[:,None]

        return ce


def convNet(output_size, im_height, im_width):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(im_height,im_width,1),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Conv2D(256, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Flatten())
    model.add(Dense(256, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(output_size, activation='sigmoid'))
    return model


def get_model(output_size=16, im_height=256, im_width=256):
    model = convNet(output_size, im_height, im_width)
    return model

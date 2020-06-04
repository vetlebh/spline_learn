
import numpy as np

import tensorflow as tf
from tensorflow_graphics.math.interpolation import bspline
from tensorflow import keras
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from tensorflow.keras.layers import Lambda, RepeatVector, Reshape, Flatten
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam



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



def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    """
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    """

    return x


def network(input_img, n_filters = 16, dropout = 0.1, batchnorm = True):

    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    p5 = MaxPooling2D((2, 2))(c5)
    p5 = Dropout(dropout)(p5)

    #outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)


    f = Flatten()(p5)

    d = Dense(16, activation='sigmoid')(f)

    outputs = d * 255.9
    model = Model(inputs=[input_img], outputs=[outputs])
    return model





def get_model(im_height=256, im_width=256, n_filters=16, dropout=0.05, batchnorm=True):
    input_img = Input((im_height, im_width, 1), name='img')
    model = network(input_img, n_filters=n_filters, dropout=dropout, batchnorm=batchnorm)
    #model.compile(optimizer=Adam(), loss=pixelizedIoU(), metrics=["accuracy"])
    return model

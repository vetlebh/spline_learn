
from google.colab import files
files.upload()

from google.colab import drive
drive.mount("/content/drive")

import tensorflow as tf
from tensorflow import keras
import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import BSpline
from cv2 import floodFill

from spline import Spline, get_spline
from network import get_model, convNet
from get_data import get_dataset, get_raw_dataset
from utils import iou

from skimage.transform import resize
from sklearn.model_selection import train_test_split

import scipy
from scipy.integrate import simps

if 'COLAB_TPU_ADDR' not in os.environ:
  print('Not connected to TPU')
else:
  print("Connected to TPU")

path = '/content/drive/My Drive/curvecnn'
model = keras.models.load_model(path)

!pip show tensorflow

gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Select the Runtime > "Change runtime type" menu to enable a GPU accelerator, ')
  print('and then re-execute this cell.')
else:
  print(gpu_info)

from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
  print('To enable a high-RAM runtime, select the Runtime > "Change runtime type"')
  print('menu, and then select High-RAM in the Runtime shape dropdown. Then, ')
  print('re-execute this cell.')
else:
  print('You are using a high-RAM runtime!')

"""### Generate Data"""

train_dataset, valid_dataset, test_dataset = get_dataset(which='colab', nr_patients=10, val_size=0.1, random_state=69)

checkpoint_path = 'weights_1.h5'
model_path = 'model_1.h5'

"""### Help Functions"""


def calc_stepsize(nd, d, p_i, delta_t):
    """
    Based on theroretical estimate, see report
    """
    return delta_t/(np.sum(np.sqrt(np.sum(np.square(p_i), axis=1))))



def floodFill_start(img, row):
    """
    Iterates outwards from start row to find point satisfied by
    ray-casting algortihm to be inside spline
    """
    if (row >= 256 or row < 0):
        return False
    next_row = int(row + 10)
    intersect = np.where(img[row, :] > 0.0)[0]
    nr_intersect = intersect.shape[0]
    if (nr_intersect > 0):
        if (intersect[0] > 0):
            return (0, row)
        else:
            for i in range (0, nr_intersect-2, 2):
                if (intersect[i+2]-intersect[i+1] > 1):
                    return (int((intersect[i+1]+intersect[i])/2), row)
        return floodFill_start(img, next_row)
    else:
        return floodFill_start(img, next_row)


def floodFill_(img, p_i):
    img = np.pad(img, 1, 'constant', constant_values=(0))
    #start_ind = floodFill_start(img, 0)
    start_ind = (0,0)
    img = img.astype('uint8')
    img = floodFill(img, None, start_ind, 1.0)[1]
    cond = np.where((img==0) | (img==1))
    img[cond] = 1-img[cond]
    img = img.astype('float32')
    #img[np.where(img > 1)] = img[np.where(img > 1)]/255.0
    img = img[1:-1, 1:-1]
    return img


def get_spline_img(p, batch_size, n):
    #time_start = time.time()
    d = 3
    nd = n+d
    rows = 256
    cols = 256

    img_batch = np.zeros((batch_size, rows, cols))

    # Iterate over all examples of batch
    for i in range(batch_size):
        p_i = p[i, :].reshape(n, 2)
        p_temp = np.zeros((nd,2))
        p_temp[:n] = p_i
        p_temp[n:] = p_i[:d]
        p_i = p_temp

        t = np.linspace(0, 1, nd+d+1)
        delta_t = 1/(nd+d)
        spline = BSpline.construct_fast(t, p_i, d, 'periodic')

        # Calculate step size
        step = calc_stepsize(nd, d, p_i, delta_t)

        # Calculate spline
        img = np.zeros((rows, cols))
        t_n = np.linspace(0, 1, int(1.0/step))

        # Find Interior Pixels
        point_prev = spline(t_n[0])
        col_prev= int(point_prev[0])
        row_prev = int(point_prev[1])
        for x in t_n:
            point = spline(x)
            col = int(point[0])
            row = int(point[1])
            if (col==col_prev and row==row_prev):
                continue
            else:
                if img[row_prev, col_prev] == 0:
                    img[row_prev, col_prev] = 1
                col_prev = col
                row_prev = row
        point = spline(t[-1])
        col = int(point[0])
        row = int(point[1])
        img[row, col] = 1

        # Flood fill spline
        img = floodFill_(img, p_i)


        # Calculate edge pixel value
        point_prev = spline(t_n[0])
        col_prev = int(point_prev[0])
        row_prev = int(point_prev[1])
        x_arr = []
        y_arr = []
        for x in t_n:

            point = spline(x)
            col = int(point[0])
            row = int(point[1])

            if (col==col_prev and row==row_prev):
                x_arr.append(point[0]-col)
                y_arr.append(point[1]-row)

            else:


                # NORTH
                if img[row_prev-1, col_prev] == 1:
                    img[row_prev, col_prev] = np.abs(scipy.integrate.trapz(y_arr, x_arr))
                # EAST
                elif img[row_prev, col_prev-1] == 1:
                    img[row_prev, col_prev] = np.abs(scipy.integrate.trapz(x_arr, y_arr))
                # SOUTH
                elif img[row_prev+1, col_prev] == 1:
                    img[row_prev, col_prev] = 1 - np.abs(scipy.integrate.trapz(y_arr, x_arr))
                # WEST
                elif img[row_prev, col_prev+1] == 1:
                    img[row_prev, col_prev] = 1 - np.abs(scipy.integrate.trapz(x_arr, y_arr))
                # NORTH WEST
                elif img[row_prev-1, col_prev+1] == 1:
                    img[row_prev, col_prev] = np.abs(scipy.integrate.trapz(y_arr, x_arr))
                # NORTH EAST
                elif img[row_prev-1, col_prev-1] == 1:
                    img[row_prev, col_prev] = np.abs(scipy.integrate.trapz(y_arr, x_arr))
                # SOUTH WEST
                elif img[row_prev+1, col_prev+1] == 1:
                    img[row_prev, col_prev] = 1-np.abs(scipy.integrate.trapz(y_arr, x_arr))
                # SOUTH EAST
                elif img[row_prev+1, col_prev-1] == 1:
                    img[row_prev, col_prev] = 1-np.abs(scipy.integrate.trapz(y_arr, x_arr))


                x_arr = []
                y_arr = []
                x_arr.append(point[0]-col)
                y_arr.append(point[1]-row)
                col_prev = col
                row_prev = row


        # Flood fill spline
        img_batch[i, :, :] = img
        #plt.imshow(img)
        #plt.show()

    return img_batch

def seg_loss(labels, predictions, p, beta, axis=0):
    #beta = 0.01
    bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    ce = tf.reduce_sum(bce(labels, predictions), axis=axis)
    #print(ce)
    #reg = beta * tf.reduce_sum(tf.math.square(tf.abs(p-(256.0/2))), axis=1)
    reg = tf.keras.losses.MSE(256.0/2, p)
    reg = tf.cast(reg, dtype='float32')
    #print(tf.math.add(ce, beta*reg))
    return tf.math.add(ce, beta*reg)
    #return reg


@tf.custom_gradient
def spline_loss(labels, predictions, dP, beta):

    labels = tf.squeeze(labels)
    p = predictions.numpy() # Transform input tensor to numpy-array
    batch_size, output_size = p.shape[0], p.shape[1]
    n = int(output_size/2)  # n is number of control points

    # Calculate batch loss
    predictions = tf.constant(get_spline_img(p, batch_size, n), dtype='float32')  # Draw spline from control points
    predictions = tf.squeeze(predictions)
    assert predictions.numpy()[np.where(predictions.numpy()>1)].size == 0
    loss = seg_loss(labels, predictions, p, beta, axis=1)

    # Custom gradient for spline loss
    def spline_grad(dLoss, dPred):

        difference = 'central'
        #dP = 1.0
        gradient = np.zeros((batch_size, output_size))
        #ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)

        # Iterate over all samples in batch
        for sample in range(batch_size):

            # Kan NONE FJERNES? UNØDVENDIG DIM
            p_sample = p[None, sample, :].copy()

             # Iterate over all control points to calculate gradient
            for i in range(output_size):

                dP_ = np.zeros((1, output_size))
                dP_[:,i] = dP


                #Forward/backward difference approximation to gradient
                if (difference == 'forward'):

                    # Check if point close to boundary
                    if (p_sample[:, i] + dP < 256.0/2):
                        sign = 1
                    else:
                        sign = -1

                    p_dP = p_sample + sign*dP_
                    img = get_spline_img(p_dP, 1, n)

                    # MÅ FJERNES HVIS VEKTORISERING
                    img = np.squeeze(img)
                    assert img[np.where(img>1)].size == 0

                    loss_dP = seg_loss(labels[sample, :, :], tf.constant(img, dtype='float32'), p_dP, beta, axis=0)
                    gradient[sample, i] = sign*(loss_dP - loss[sample])/dP


                # Central difference approximation to gradient
                elif (difference == 'central'):

                    if (p_sample[:, i] + dP < 256.0):
                        p_dp_forw = p_sample + dP_
                    else:
                        p_dp_forw = p_sample

                    if (p_sample[:, i] - dP >= 0.0):
                        p_dp_back = p_sample - dP_
                    else:
                        p_dp_back = p_sample

                    img_forw = get_spline_img(p_dp_forw, 1, n)
                    img_back = get_spline_img(p_dp_back, 1, n)

                    img_forw = np.squeeze(img_forw)
                    assert img_forw[np.where(img_forw>1)].size == 0
                    img_back = np.squeeze(img_back)
                    assert img_back[np.where(img_back>1)].size == 0

                    loss_dP_forw = seg_loss(labels[sample, :, :], tf.constant(img_forw, dtype='float32'), p_dp_forw, beta, axis=0)
                    loss_dP_back = seg_loss(labels[sample, :, :], tf.constant(img_back, dtype='float32'), p_dp_back, beta, axis=0)
                    #print(loss_dP_forw, loss_dP_back)
                    gradient[sample, i] = (loss_dP_forw - loss_dP_back) / (2*dP)


                #ta = ta.write(sample, sign*(loss_dP - loss[sample])/dP)

        gradient = tf.constant(gradient, dtype='float32')
        #gradient = ta
        #print(gradient)
        return None, gradient, None, None

    return (loss, predictions), spline_grad


@tf.custom_gradient
def spline_loss_test(labels, predictions):

    #loss =  beta * tf.reduce_sum(tf.math.square(tf.abs(p-(256.0/2))), axis=1)
    loss = tf.keras.losses.MSE(256.0/2, predictions)
    #loss = tf.cast(loss, dtype='float32')

    def grad_fn(dLab):
        grad = -2*(256.0/2 - predictions)
        #grad = tf.cast(grad, dtype='float32')
        return None, grad

    return loss, grad_fn
    #return loss


# Train the model
#@tf.function # Speeds things up
# https://www.tensorflow.org/tutorials/customization/performance
def model_train(features, labels, dP, beta):

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Define the GradientTape context
    with tf.GradientTape() as tape:

        # Get the probabilities
        predictions = 255.9*model(features, training=True)

        # Calculate the loss
        loss, predictions = spline_loss(labels, predictions, dP, beta)


    # Get the gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    #print(gradients)

    # Update the weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Update the loss and accuracy
    labels = tf.squeeze(labels)
    train_loss(loss)
    train_iou(labels, predictions)


def model_validate(features, labels, dP, beta):
    predictions = 255.9*model(features, training=False)
    v_loss, predictions = spline_loss(labels, predictions, dP, beta)
    labels = tf.squeeze(labels)
    valid_loss(v_loss)
    valid_iou(labels, predictions)

"""### Model

### Train
"""

train_loss = tf.keras.metrics.Mean(name="train_loss")
valid_loss = tf.keras.metrics.Mean(name="test_loss")

# Specify the performance metric
train_iou = tf.keras.metrics.MeanIoU(num_classes=2, name="train_iou")
valid_iou = tf.keras.metrics.MeanIoU(num_classes=2, name="valid_iou")

best_loss = 99999
num_epochs = 50
beta = 0
dP = 1.25

loss_arr = []
val_loss_arr =[]

# Train the model for num_epochs
for epoch in range(num_epochs):

    model_save_name = 'curvecnn35'
    path = f"/content/drive/My Drive/{model_save_name}"
    model.save(path)

    np.save('/content/drive/My Drive/val_loss_arr35', val_loss_arr)
    np.save('/content/drive/My Drive/loss_arr35', loss_arr)

    # Run the model through train and validation sets respectively
    for (features, labels) in train_dataset:
        model_train(features, labels, dP, beta)


    for valid_features, valid_labels in valid_dataset:
        model_validate(valid_features, valid_labels, dP, beta)


    # Grab the results
    (loss, iou) = train_loss.result(), train_iou.result()
    (val_loss, val_iou) = valid_loss.result(), valid_iou.result()

    loss_arr.append(loss.numpy())
    val_loss_arr.append(val_loss.numpy())



    # Clear the current state of the metrics
    train_loss.reset_states(), train_iou.reset_states()
    valid_loss.reset_states(), valid_iou.reset_states()


    # Local logging
    template = "Epoch {}, loss: {:.3f}, iou: {:.3f}, val_loss: {:.3f}, val_iou: {:.3f}"
    print (template.format(epoch+1,
                        loss,
                        iou,
                        val_loss,
                        val_iou))


    if (val_loss < best_loss):
        best_loss = val_loss
        model.save_weights('model.h5')
        print(f'New low val_loss. Model weights saved.')

    if ((epoch+1)%5 == 0):
        preds = model.predict(test_dataset)*259.9
        plt.imshow(get_spline(preds[0, :].reshape(8,2)), origin='lower')
        plt.show()

    if (epoch==5):
        beta = 0

    if ((epoch+1)%10 == 0):
        dP = dP/2.0
        print(f'New dP: {dP}')

model.save('colab1')

!zip -r /content/file.zip /content/colab1_epoch19/

from google.colab import files
files.download('example.txt')

"""### Evaluate"""

preds = model.predict(test_dataset)*259.9

i = 0

print(preds[i, :])

spline = Spline(preds[i, :])
spline.drawCurve()

i=4
plt.imshow(get_spline(preds[i, :].reshape(8,2)))
plt.imshow()

preds = model.predict(X_test)
print(np.unique(preds))

print(preds[0, :])

spline = Spline(preds[0, :])
spline.drawCurve()
spline.get_spline(flood=True)

plt.imshow(np.squeeze(y_test[1,:]))

a = preds[None, 1, :].copy()
print(a.shape)

valid_acc(y_test[1,:], get_spline_img(preds[None, 1, :], 1, 8))

preds[1, :]

a = np.ones(5)
a[0] = 0
a[1] = 0.2
a[2] = 10
b = np.zeros(5)
bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
print(bce(a,b))
assert a[np.where(a>1)].size == 0

model.layers[10].get_weights()

p = np.ones((2, 10))
pd = tf.reduce_sum(tf.abs(p-256/2), axis=1)
pd.dtype

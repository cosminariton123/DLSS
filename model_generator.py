from keras.layers import Flatten, ReLU, Conv2D, Dense, Conv2DTranspose
from keras.layers import Dropout, BatchNormalization, RandomCrop, RandomRotation, GaussianNoise, MaxPool2D
from keras import optimizers, Model
import tensorflow as tf

from config import INPUT_SIZE
from custom_metrics import CUSTOM_METRICS
from keras_tuner import HyperParameters

from util import INPUT_SIZE_NUMPY


def make_model(hp: HyperParameters):

    input = tf.keras.Input(shape=INPUT_SIZE_NUMPY)


    layer = input

    gaussian_noise = hp.Boolean(f"Gaussian Noise")
    with hp.conditional_scope("Gaussian Noise", True):
        if gaussian_noise:
            gaussian_noise_stddev = hp.Float(f"Gaussian Noise value", min_value=0.001, max_value=0.1, step=10, sampling="log")
            layer = GaussianNoise(gaussian_noise_stddev)(layer)

    layer = tf.clip_by_value(layer, -1, 1)

    nr_ConvLayers = 20

    for i in range(nr_ConvLayers):
        nr_filters = hp.Int(f"Nr filters ConvLayer {i}", min_value=8, max_value=64, step=2, sampling="log")

        if i < nr_ConvLayers - 1:
            layer = Conv2DTranspose(nr_filters, 7)(layer)
        else:
            layer = Conv2DTranspose(nr_filters, 2)(layer)

        bn = hp.Boolean(f"BN ConvLayer {i}")
        if bn:
            layer = BatchNormalization()(layer)


        dp = hp.Boolean(f"Dropout ConvLayer {i}")
        if dp:
            dp_value = hp.Float(f"Dropout value ConvLayer {i}", min_value=0.1, max_value=0.5, step=0.1, sampling="linear")
            layer = Dropout(dp_value)(layer)

        layer = ReLU()(layer)

    layer = Conv2D(3, 1, activation="tanh", dtype = tf.float32)(layer)

    learning_rate = hp.Float(f"Learning rate", min_value=1e-5, max_value=1e-1, step=10, sampling="log")

    optimizer = optimizers.Nadam(learning_rate=learning_rate)
    optimizer =  tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic = True)
    
    model = tf.keras.Model(inputs = input, outputs = layer)

    loss = hp.Choice("loss", ["mse", "mae"])

    model.compile(loss=loss, optimizer=optimizer, metrics=CUSTOM_METRICS)

    return model
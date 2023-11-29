import os

import tensorflow as tf

from config import REDUCE_LR_COOLDOWN, REDUCE_LR_PATIENCE, REDUCE_LR_MIN_DELTA, REDUCE_LR_MIN_LR, REDUCE_LR_FACTOR
from config import HISTOGRAM_FREQ, WRITE_GRAPHS, WRITE_IMAGES, WRITE_STEPS_PER_SECOND, PROFILE_BATCH
from config import EARLY_STOPPING_PATIENTE_IN_EPOCHS, EARLY_STOPPING_MIN_DELTA


def generate_callbacks(save_path):
    callbacks = list()
    
    callbacks.append(tf.keras.callbacks.TerminateOnNaN())

    callbacks.append(tf.keras.callbacks.EarlyStopping(
        min_delta = EARLY_STOPPING_MIN_DELTA,
        patience = EARLY_STOPPING_PATIENTE_IN_EPOCHS,
        verbose = 1
    ))

    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
        factor = REDUCE_LR_FACTOR,
        patience = REDUCE_LR_PATIENCE,
        verbose = 1,
        min_delta = REDUCE_LR_MIN_DELTA,
        cooldown = REDUCE_LR_COOLDOWN,
        min_lr = REDUCE_LR_MIN_LR,
        mode="min"
    ))

    log_dir = os.path.join(save_path, "logs")

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    callbacks.append(tf.keras.callbacks.TensorBoard(
        log_dir = log_dir,
        histogram_freq = HISTOGRAM_FREQ,
        write_graph = WRITE_GRAPHS,
        write_images = WRITE_IMAGES,
        write_steps_per_second = WRITE_STEPS_PER_SECOND,
        profile_batch = PROFILE_BATCH
    ))

    return callbacks


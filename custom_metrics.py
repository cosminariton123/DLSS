import tensorflow as tf

CUSTOM_METRICS = ["mse", "mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")]

CUSTOM_OBJECTS = CUSTOM_METRICS

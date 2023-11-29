from config import LIMIT_GPU_MEMORY_GROWTH, MIXED_PRECISION_16
from limit_gpu_memory_growth import limit_gpu_memory_growth
from keras import mixed_precision

if LIMIT_GPU_MEMORY_GROWTH: limit_gpu_memory_growth()
if MIXED_PRECISION_16 is True: mixed_precision.set_global_policy('mixed_float16')

import os
import numpy as np
import keras_tuner
from keras import Model


from data_loader import TrainingGenerator, PredictionsGenerator, load_samples
from preprocessing import preprocess_image_training, preprocess_image_predicting
from model_generator import make_model
from callbacks import generate_callbacks
from config import TRAINING_BATCH_SIZE, EPOCHS, MAX_TRIALS, KFold_k, TRAIN_SIZE_SPLIT, OBJECTIVE
from CVTuner import CVTuner
from prediction import make_prediction

from sklearn.model_selection import train_test_split

import time



def main():

    DATASET_DIR = "dataset"
    PREDICTION_DIR = "prediction"

    filepaths = load_samples(DATASET_DIR)
    
    filepaths, filepaths_test = train_test_split(filepaths, train_size=TRAIN_SIZE_SPLIT, random_state=1)
    filepaths = np.array(filepaths)
    filepaths_test = np.array(filepaths_test)

    save_path = "WORK_DIR"

    tuner = CVTuner(
    hypermodel=make_model,
    oracle=keras_tuner.oracles.BayesianOptimizationOracle(
        objective=OBJECTIVE,
        max_trials=MAX_TRIALS,
        max_consecutive_failed_trials=10**10
        ),
        directory=save_path,
        )

    tuner.search(
        filepaths,
        filepaths,
        K_fold_k = KFold_k, 
        batch_size=TRAINING_BATCH_SIZE, 
        epochs=EPOCHS,
        callbacks = generate_callbacks(save_path)
        )

    time.sleep(5)

    print(f"\n\n\n########################################################\n")
    tuner.results_summary(1)
    print(f"########################################################\n\n\n")

    best_model: Model = tuner.get_best_models()[0]

    best_model.summary()

    prediction_data_generator = PredictionsGenerator(filepaths=filepaths_test,
                batch_size=1,
                preprocessing_procedure=preprocess_image_predicting,
    )

    loss, mse, mae, rmse = best_model.evaluate(prediction_data_generator)

    print(f"\n\n\n########################################################\nTEST_SET_RESULTS\nLOSS: {loss}\nMSE:{mse}\nMAE:{mae}\nRMSE:{rmse}\n########################################################\n\n\n")

    make_prediction(best_model, filepaths_test, PREDICTION_DIR)

if __name__ == "__main__":

    main()

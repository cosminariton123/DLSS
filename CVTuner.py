import keras_tuner
import numpy as np
from sklearn import model_selection
from keras_tuner.engine import tuner_utils
import copy
from tqdm import tqdm

from data_loader import TrainingGenerator
from preprocessing import preprocess_image_training

class CVTuner(keras_tuner.Tuner):
    def run_trial(self, trial, x, y, *args, K_fold_k, batch_size=32, epochs=1, **kwargs):

        model_checkpoint = tuner_utils.SaveBestEpoch(
            objective=self.oracle.objective,
            filepath=self._get_checkpoint_fname(trial.trial_id),
        )
        original_callbacks = kwargs.pop("callbacks", [])

        histories = []

        cv = model_selection.KFold(K_fold_k, shuffle=True)

        pbar = tqdm(desc="CrossValidation", total=K_fold_k)

        for idx, (train_indices, test_indices) in enumerate(cv.split(x)):
            x_train, x_test = x[train_indices], x[test_indices]
            model = self.hypermodel.build(trial.hyperparameters)

            copied_kwargs = copy.copy(kwargs)
            callbacks = self._deepcopy_callbacks(original_callbacks)
            self._configure_tensorboard_dir(callbacks, trial, idx)
            callbacks.append(tuner_utils.TunerCallback(self, trial))
            # Only checkpoint the best epoch across all executions.
            callbacks.append(model_checkpoint)
            copied_kwargs["callbacks"] = callbacks

            model.summary()

            training_data_generator = TrainingGenerator(filepaths=x_train,
                batch_size=batch_size,
                preprocessing_procedure=preprocess_image_training,
                shuffle=True
            )

            validation_data_generator = TrainingGenerator(filepaths=x_test,
                batch_size=batch_size,
                preprocessing_procedure=preprocess_image_training,
                shuffle=True
            )

            history = model.fit(training_data_generator, validation_data=validation_data_generator
            , epochs=epochs, shuffle=False, **copied_kwargs)
            pbar.update(1)

            histories.append(history)

        return histories

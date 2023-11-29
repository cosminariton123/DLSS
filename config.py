import cv2
from PIL import Image
import multiprocessing


#DATA CONFIG
INPUT_SIZE = (230//2, 230//2, 3)
GROUND_TRUTH_SIZE = (230, 230, 3)

#ML CONFIG
EPOCHS = 70
EARLY_STOPPING_PATIENTE_IN_EPOCHS = 11
EARLY_STOPPING_MIN_DELTA = 0
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 4
REDUCE_LR_COOLDOWN = 1
REDUCE_LR_MIN_DELTA = 0.000_01
REDUCE_LR_MIN_LR = 1e-6
TRAINING_BATCH_SIZE = 8 #Should be multiple of 8(even better 128 for TPUs) for better efficiency
PREDICTION_BATCH_SIZE = 1

TRAIN_SIZE_SPLIT = 0.9
MAX_TRIALS = 500
KFold_k = 3
OBJECTIVE = "val_mae"

    #Use this if you want to use your computer for something else
    #and performance of the pc is hindered by training
    #Fragmentation of memory will be higher
LIMIT_GPU_MEMORY_GROWTH = True
    #Use this if you want to have lower floating point precission(has almost no effect on loss),
    #but use less memory 
    #and compute faster for graphics cards with compute capability above 7
MIXED_PRECISION_16 = True

#TENSORBOARD CONFIG
HISTOGRAM_FREQ = 1
WRITE_GRAPHS = True
WRITE_IMAGES = False
WRITE_STEPS_PER_SECOND = True
PROFILE_BATCH = 0


#PREPROCESSING CONFIG
INTERPOLATION_RESIZE = cv2.INTER_CUBIC
INTERPOLATION_ROTATE = Image.Resampling.BICUBIC
NOISE_PERCENTAGE = 0.1
NOISE_MEAN = 0
NOISE_DEVIATION = 20
MAX_DEGREE_OF_LEFT_RIGHT_ROTATION = 30
MAX_BRIGHTNESS_AUGM_COEFICIENT = 0.2 #0 is unchanged, 1.2 is up to 20% more or less brightness

#OPTIMIZATIONS CONFIG
NR_OF_PROCESSES = 1#multiprocessing.cpu_count()

#PREPROCESSING INFERENCE CONFIG
RESIZE_TO_TRAINING_SIZE = True

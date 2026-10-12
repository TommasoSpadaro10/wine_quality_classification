import torch

DATA_DIR = 'data'
PLOTS_DIR = 'plots'
MODELS_DIR = 'models'
TENSORBOARD_DIR = 'runs'
MODEL_VERSION = '0.1.0'
EPOCHS = 20000
BATCH_SIZE = 64
EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_PATIENCE = 2500
EARLY_STOPPING_MIN_ACCURACY = 0.7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.2
FEATURES = [
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'pH',
    'sulphates',
    'alcohol',
    'acidity_ratio',
    'sulfur_ratio',
    'alcohol_density',
    'total_acidity'
]
LABEL = 'quality'
NUM_CLASSES = 10
PENALTY_BAD_TO_GOOD = 12.0
PENALTY_GOOD_TO_BAD = 19.5
QUALITY_THRESHOLD = 5
RANDOM_STATE = 42
LEARNING_RATE = 0.003
ADAM_BETAS = (0.9, 0.999)
WEIGHT_DECAY = 1e-4
ORTHO_LAMBDA = 0.2
N_CLUSTERS = 2
NCL_LAMBDA = 0.5
PLAUSIBLE_RANGES = {
    'fixed acidity':        (3.0, 16.0),
    'volatile acidity':     (0.0, 1.2),
    'citric acid':          (0.0, 1.0),
    'residual sugar':       (0.0, 8.0),
    'chlorides':            (0.0, 0.2),
    'free sulfur dioxide':  (1.0, 72.0),
    'total sulfur dioxide': (1.0, 200.0),
    'density':              (0.985, 1.010),
    'pH':                   (2.5, 4.5),
    'sulphates':            (0.0, 1.5),
    'alcohol':              (8.0, 15.5),
}

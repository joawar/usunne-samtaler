BINARY = True
EVALUATE_DURING_TRAINING = False
EVAL_STRAT = 'no'
SAVE_STRAT = 'no'
LOAD_BEST_LAST = False
DO_LOWER_CASE = False


UNDERSAMPLING = True
OVERSAMPLING = False

ENG_OVERSAMPLING = False
PURE_ENG_OVERSAMPLE = False
N_ENG_DATAPOINTS = 300


COMBINE_TRAIN_VAL = True
CROSS_VALIDATION = True
SAVE_MODEL = False

# LOG_INTERVAL = 50
N_LOGS = 22
MAX_LEN = 210
EPOCHS = 3
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
WARMUP_PROPORTION = 0.1
SEED = 42

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.20

CHARACTERISTIC = 7  # see make_binary_df in dataset.py
CHARACTERISTIC_NAME = 'unhealthy'

# cross validation
N_SPLITS = 5
N_REPEATS = 1

# LOSS_WEIGHTS = [0.15, 0.85]
LOSS_WEIGHTS = None
TEST = 1

MODEL_STR = 'NbAiLab/nb-bert-base'
# MODEL_STR = 'models/nb-bert-unhealthy-trans'
# MODEL_STR = 'models/m-bert-unhealthy'
# MODEL_STR = 'models/nb-bert-unhealthy-UCC'
# MODEL_STR = 'bert-base-multilingual-cased'
# MODEL_STR = 'bert-base-cased'
# MODEL_STR = 'ltgoslo/norbert'
# MODEL_STR = 'models/norbert-unhealthy-UCC'

OUTPUT_DIR = 'models'

RUN_NAME = f'm-bert-{CHARACTERISTIC_NAME}-undersampling'
# RUN_NAME = f'nb-bert-{CHARACTERISTIC_NAME}-cont-undersampling'
# RUN_NAME = f'm-bert-{CHARACTERISTIC_NAME}-cont-undersampling'

# TRAIN_PATH = 'data/norwegian/translated.csv'
TRAIN_PATH = 'data/train_HU.csv'
# TRAIN_PATH = 'data/norwegian/train.csv'
# TRAIN_PATH = 'data/UCC/full.csv'

SAVE_DIR = f'{OUTPUT_DIR}/{RUN_NAME}'

METRIC_FILE = f'out/metrics/{RUN_NAME}'

VAL_PATH = 'data/val_HU.csv'
# VAL_PATH = 'data/norwegian/val.csv'
# VAL_PATH = 'data/UCC/val.csv'


if BINARY:
    COMPUTE_OBJECTIVE = 'PR_AUC'
else:
    COMPUTE_OBJECTIVE = 'Hamming_score'

if BINARY:
    N_LABELS = 2
else:
    N_LABELS = 7

"""
Configuration file for the Cancer Detection Project.
Stores paths, hyperparameters, and other global settings.
All paths are constructed relative to the project root.
"""
import os
import logging

# --- Project Root Directory ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- Data Paths ---
BASE_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'lung_colon_image_set')
COLON_DATA_DIR = os.path.join(BASE_DATA_DIR, 'colon_image_sets')
LUNG_DATA_DIR = os.path.join(BASE_DATA_DIR, 'lung_image_sets')

# --- Output Directories ---
MODEL_DIR = os.path.join(PROJECT_ROOT, 'trained_models')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

COLON_REPORT_DIR = os.path.join(REPORTS_DIR, 'colon')
LUNG_REPORT_DIR = os.path.join(REPORTS_DIR, 'lung')

TENSORBOARD_LOG_DIR_COLON = os.path.join(LOGS_DIR, 'fit', 'colon')
TENSORBOARD_LOG_DIR_LUNG = os.path.join(LOGS_DIR, 'fit', 'lung')

APP_LOG_FILE = os.path.join(LOGS_DIR, 'application.log')

# --- Ensure Directories Exist ---
def create_directories():
    dirs_to_create = [
        MODEL_DIR, REPORTS_DIR, LOGS_DIR,
        COLON_REPORT_DIR, LUNG_REPORT_DIR,
        TENSORBOARD_LOG_DIR_COLON, TENSORBOARD_LOG_DIR_LUNG
    ]
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)

# --- Image Processing Parameters ---
TARGET_IMAGE_WIDTH = 128
TARGET_IMAGE_HEIGHT = 128
TARGET_IMAGE_SIZE = (TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

# --- Dataset Split Parameters ---
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
RANDOM_STATE = 42

# ======================================================
# ✅ IMPORTANT FIX: USE .h5 (STABLE FORMAT)
# ======================================================

# --- Colon Model Config ---
COLON_MODEL_NAME = 'colon_cancer_model.h5'   # 🔧 changed
COLON_MODEL_PATH = os.path.join(MODEL_DIR, COLON_MODEL_NAME)
COLON_HISTORY_PATH = os.path.join(MODEL_DIR, 'colon_model_history.pkl')
COLON_LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'colon_label_encoder.pkl')
COLON_CLASSES = ['colon_aca', 'colon_n']
COLON_NUM_CLASSES = len(COLON_CLASSES)
COLON_DEFAULT_EPOCHS = 30
COLON_DEFAULT_BATCH_SIZE = 32

# --- Lung Model Config ---
LUNG_MODEL_NAME = 'lung_cancer_model.h5'     # 🔧 changed
LUNG_MODEL_PATH = os.path.join(MODEL_DIR, LUNG_MODEL_NAME)
LUNG_HISTORY_PATH = os.path.join(MODEL_DIR, 'lung_model_history.pkl')
LUNG_LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'lung_label_encoder.pkl')
LUNG_CLASSES = ['lung_aca', 'lung_n', 'lung_scc']
LUNG_NUM_CLASSES = len(LUNG_CLASSES)
LUNG_DEFAULT_EPOCHS = 25
LUNG_DEFAULT_BATCH_SIZE = 32

# --- Training Callbacks Parameters ---
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 7
REDUCE_LR_FACTOR = 0.2
MONITOR_METRIC = 'val_accuracy'
MIN_DELTA_EARLY_STOPPING = 0.001

# --- Logging Configuration ---
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'

def get_logger(name, level=LOG_LEVEL, log_file=APP_LOG_FILE):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)

        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(ch)

        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(fh)

    return logger

# --- Create directories on import ---
create_directories()

if __name__ == '__main__':
    logger_instance = get_logger(__name__)
    logger_instance.info("Configuration loaded successfully.")

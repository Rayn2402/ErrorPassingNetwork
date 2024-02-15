"""
Description : Stores a custom enumeration of the important paths within the project
"""

from os.path import dirname, join
from src.data.extraction import constants as cst


class Paths:
    """
    Paths of important directories and files
    """
    PROJECT_DIR: str = dirname(dirname(__file__))
    CHECKPOINTS: str = join(PROJECT_DIR, "checkpoints")
    DATA: str = join(PROJECT_DIR, "data")
    SCRIPTS: str = join(PROJECT_DIR, "scripts")
    EXPERIMENTS_SCRIPTS: str = join(SCRIPTS, "experiments")
    POST_ANALYSES_SCRIPTS: str = join(SCRIPTS, "post_analyses")
    UTILS_SCRIPTS: str = join(SCRIPTS, "utils")
    VO2_DATASET_CSV = join(DATA, "vo2_dataset_random.csv")
    HYPERPARAMETERS: str = join(PROJECT_DIR, "hps")
    MASKS: str = join(PROJECT_DIR, "masks")
    VO2_MASK: str = join(MASKS, "vo2_mask.json")
    MODELS: str = join(PROJECT_DIR, "models")
    RECORDS: str = join(PROJECT_DIR, "records")
    CLEANING_RECORDS: str = join(RECORDS, "cleaning")
    CSV_FILES: str = join(RECORDS, "csv")
    DESC_RECORDS: str = join(RECORDS, "descriptive_analyses")
    DESC_CHARTS: str = join(DESC_RECORDS, "charts")
    DESC_STATS: str = join(DESC_RECORDS, "stats")
    EXPERIMENTS_RECORDS: str = join(RECORDS, "experiments")
    FIGURES_RECORDS: str = join(RECORDS, "figures")
    TUNING_RECORDS: str = join(RECORDS, "tuning")


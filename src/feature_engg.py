import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler

log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")

log_file_path = os.path.join(log_dir,"data_preprocessing.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


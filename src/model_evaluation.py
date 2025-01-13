import pandas as pd
import numpy as np
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,f1_score,roc_auc_score


log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger("model_trainer")
logger.setLevel("DEBUG")

log_file_path = os.path.join(log_dir,"model_trainer.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

def model_prediction(model_obj_path:str,test_data_path:str):
    try:
        with open(model_obj_path,"rb") as f:
            model = pickle.load(f)

        X_test = pd.read_csv(test_data_path)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)


        return y_pred
    except Exception as e:
        print(e)
        logger.info("Unexcepected eroor -",e)
    
def evaluation(y_test,y_pred):
    try:
        accuracy = accuracy_score(y_test,y_pred)
        Roc_Auc_score = roc_auc_score(y_test,y_pred)
        print(f"accuracy score - {accuracy}")
        print(f"roc auc score - {Roc_Auc_score}")

        logger.info(f"accuracy score - {accuracy}")
        logger.info(f"roc auc score - {Roc_Auc_score}")

    except Exception as e:
        print(e)
        logger.info("Unexcepected eroor -",e)

def main():
    try:
        model_obj_path = "/media/minato/Local Disk/My Stuff/Data Science/basic_projects/MLOPS-proj/model_objects/models/rfc.pkl"
        test_data_path = "/media/minato/Local Disk/My Stuff/Data Science/basic_projects/MLOPS-proj/data/scaled_data/X_test_scl.csv"
        y_test_path = "/media/minato/Local Disk/My Stuff/Data Science/basic_projects/MLOPS-proj/data/labels/y_test.csv"

        y_test = pd.read_csv(y_test_path)
        y_test = np.array(y_test)
        y_pred = model_prediction(model_obj_path,test_data_path)

        evaluation(y_test,y_pred)

    except Exception as e:
        print(e)
        logger.info("Unexcepected eroor -",e)

if __name__ == "__main__":
    main()
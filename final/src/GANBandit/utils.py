import logging
import random
import torch
import numpy as np

from sklearn.metrics import mean_squared_error

class Report():
    
    def __init__(self, X, y):
        self.gt = None
        self.predict = None
        self.back = None
        self.arm_loss = {}
        self.local_loss = None
        self.global_loss = None
        self.X = X
        self.y = y
        
    def evaluate_pred(self, pred, n=None):
        self.predict = pred
        self.pred_idx_list = []
        for i, j in zip(pred, self.y):
            idx = np.argmax(i)
            self.pred_idx_list.append(j[idx])
        self.global_pred_score = np.mean(self.pred_idx_list)
        self.local_pred_score = np.mean(self.pred_idx_list[:n])
        
        if len(pred) == len(self.y):
            self.global_loss = mean_squared_error(y, pred)
        self.local_loss = mean_squared_error(self.y[:n, :], pred[:n])
        
    def evaluate_back(self, back, n=None):
        self.back = back
        self.back_idx_list = []
        
        for i, j in zip(back, self.y):
            self.back_idx_list.append(j[i])
        self.global_back_score = np.mean(self.back_idx_list)
        self.local_back_score = np.mean(self.back_idx_list[:n])
        

def set_seed(seed: int):
    logging.info(f"Set seed {seed}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_logging(logging_level="INFO"):
    logging_level = logging_level.upper()
    numeric_level = getattr(logging, logging_level, None)

    if not isinstance(numeric_level, int):
        raise Exception(f"Invalid log level: {numeric_level}")

    logging.basicConfig(
        level=numeric_level,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="[%(asctime)s] [%(process)s] [%(levelname)s] [%(module)s]: #%(funcName)s @%(lineno)d: %(message)s",
    )
    logging.info(f"Logging level: {logging_level}")


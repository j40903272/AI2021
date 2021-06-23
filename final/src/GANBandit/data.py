import re
import logging
import pandas as pd
import numpy as np
from collections import defaultdict

from sklearn.utils import shuffle
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from GANBandit.utils import set_seed


class Rec():
    
    def __init__(self, path, frac=None, duplicate=1):
        logging.info("reading data")
        df = pd.read_csv("../../ml-latest-small/ratings.csv")
        
        if frac is not None:
            if frac < 1:
                df = df.sample(frac=frac)
            else:
                df = df.sample(frac=frac, replace=True)
        
        logging.info(f"""
            df length: {len(df)}
            user: {df.userId.nunique()}
            item: {df.movieId.nunique()}
            rate: {sorted(df.rating.unique())}
        """)
        
        self.n_item = df.movieId.nunique()
        self.n_user = df.userId.nunique()
        
        # encode label
        uid_encoder = LabelEncoder()
        df.userId = uid_encoder.fit_transform(df.userId)
        mid_encoder = LabelEncoder()
        df.movieId = mid_encoder.fit_transform(df.movieId)
        
        rating_dict = self.create_rate_mapping(df)
        self.rating_dict = rating_dict
        
        X, y = self.rate_to_click(rating_dict, duplicate)
        X, y = shuffle(X, y)
        self.X, self.y = X, y
        logging.info(f"Load dataset complete. X: {X.shape}, y: {y.shape}")

    def create_rate_mapping(self, df):
        logging.info("create rating map")
        rating_dict = defaultdict(lambda : np.zeros(self.n_item))
        for uid, mid, rate in df[['userId', 'movieId', 'rating']].values:
            uid, mid = int(uid), int(mid)
            rating_dict[uid][mid] = rate
        return rating_dict

    def rate_to_click(self, rating_dict, duplicate):
        logging.info("expand rate to click")
        X = []
        y = []
        for uid, vec in rating_dict.items():
            tmp = []
            for rate in vec:
                rate = int(rate * 2)
                cat = [1] * rate + [0] * (10 - rate)
                tmp.append(cat)

            tmp = np.array(tmp).T
            y.append(tmp)
            X.extend([uid]*10)

        X = np.array(X).reshape(-1, 1)
        y = np.concatenate(y, axis=0)
        
        # duplicate
        X = np.concatenate([X] * duplicate)
        y = np.concatenate([y] * duplicate)
        
        return X, y


class Sync():
    
    def __init__(self, n_sample, n_dim, n_class, f=None, seed=87):
        logging.info("reading data")
        if f is None:
            f = lambda x: (x * np.cos(x)) + 0.25*x

        set_seed(seed)
        action = np.linspace(-9.9, 9.9, n_class)
        X = np.random.randn(n_sample, n_dim)
        y = []
        for context in X:
            proba = f(action - context.sum()) / 50
            y.append((proba > np.random.rand(n_class)).astype(int))
        
        y = np.array(y)
        X, y = shuffle(X, y)
        self.X, self.y = X, y
        logging.info(f"Load dataset complete. X: {X.shape}, y: {y.shape}")
        
        
class Bibtext():
    
    def __init__(self, path, dim=100, duplicate=1):
        logging.info("reading data")
        X, y = self.parse_data("../../Bibtex/Bibtex_data.txt")
        X = X[:, :dim]
        y = y.astype(float)
        
        # duplicate
        X = np.concatenate([X] * duplicate)
        y = np.concatenate([y] * duplicate)
        
        X, y = shuffle(X, y)
        self.X, self.y = X, y
        logging.info(f"Load dataset complete. X: {X.shape}, y: {y.shape}")
    
    def parse_data(self, filename):
        with open(filename, "rb") as f:
            infoline = f.readline()
            infoline = re.sub(r"^b'", "", str(infoline))
            n_features = int(re.sub(r"^\d+\s(\d+)\s\d+.*$", r"\1", infoline))
            features, labels = load_svmlight_file(f, n_features=n_features, multilabel=True)
        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(labels)
        features = np.array(features.todense())
        features = np.ascontiguousarray(features)
        return features, labels

if __name__ == "__main__":
    rec_data = Rec("../ml-latest-small/ratings.csv")
    bibtext_data = Bibtext("../Bibtex/Bibtex_data.txt")
    sync_data = Sync(1000, 2, 100)
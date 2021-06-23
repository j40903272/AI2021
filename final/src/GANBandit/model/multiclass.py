import numpy as np

from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

from GANBandit.model.base import Base, _ZeroPredictor, _OnePredictor
from GANBandit.model.neural import Neural
from GANBandit.utils import Report


class MultiClass(Base):
    
    def __init__(self, nchoices, base=None, epsilon=0.0, name=None, seed=87):
        
        super().__init__(nchoices, name, seed)
        
        if not base:
            self.base = LogisticRegression(solver='lbfgs', warm_start=True, random_state=87)
        else:
            self.base = base

        self.model = OneVsRestClassifier(self.base, n_jobs=1)
        self.alpha = np.ones(nchoices) / nchoices
        self.beta = np.ones(nchoices)
        self.thr = np.ones(nchoices) * 3
        self.beta_counters = np.zeros((3, nchoices))
        self.force_fit = False
        self.force_counters = False
        self.epsilon = epsilon

    def input_trans(self, X):
        return OneHotEncoder(categories=[np.arange(9724)]).fit_transform(X)
    
    def predict(self, local_X):
        preds = self.predict_proba(local_X)
        self.evaluate(self.report_list[-1], preds, local_X)
        for choice in range(self.nchoices):
            if not isinstance(self.algos[choice], OneVsRestClassifier):
                preds[:, choice] = self.algos[choice].predict_proba(local_X)[:, 1]
        
        n = local_X.shape[0]
        preds = preds[:n]
        preds = np.argmax(preds, axis=1).astype(int)
        for e, i, j in zip(range(n), np.random.random(n), np.random.randint(0, self.nchoices, n)):
            if i < self.epsilon:
                preds[e] = j
        return preds
    
    def fit(self, X, a, r, warm_start=False):
        X = self.input_trans(X)
        Y = np.zeros((X.shape[0], self.nchoices))
        for e, (i, j) in enumerate(zip(a, r)):
            Y[e][i] = j
        self.model.fit(X, Y)  # multiclass training
        self.algos = [0] * self.nchoices
        for choice in range(self.nchoices):
            this_choice = (a == choice)
            yclass = r[this_choice]
            n_pos = (yclass > 0.).sum()
            
#             if (n_pos < self.thr[choice]) or ((yclass.shape[0] - n_pos) < self.thr[choice]):
#                 if not self.force_fit:
#                     self.algos[choice] = _BetaPredictor(self.alpha[choice] + n_pos,
#                                                         self.beta[choice] + yclass.shape[0] - n_pos,
#                                                         choice)
#                     continue
            if n_pos == 0:
                if not self.force_fit:
                    self.algos[choice] = _ZeroPredictor()
                    continue
            if n_pos == yclass.shape[0]:
                if not self.force_fit:
                    self.algos[choice] = _OnePredictor()
                    continue

            self.algos[choice] = self.model
            
        report = Report(self.X, self.y)
        self.arm_eval(report, X, a, r)
        self.report_list.append(report)
        
    def predict_proba(self, X):
        if not isinstance(self.base, Neural):
            X = self.input_trans(X)
        return self.model.predict_proba(X)

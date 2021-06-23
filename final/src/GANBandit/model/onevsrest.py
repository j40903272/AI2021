from joblib import Parallel, delayed
import time
from copy import deepcopy

from sklearn.linear_model import LogisticRegression

from .base import Base


class OneVsRest(Base):
    
    def __init__(self, nchoices, epsilon=0.0, base=None, name=None, seed=87):
        
        super().__init__(nchoices, name, seed)
        
        if base is None:
            self.base = LogisticRegression(solver='lbfgs', warm_start=True, random_state=seed)
        else:
            self.base = base

        self.alpha = np.ones(self.nchoices) / self.nchoices
        self.beta = np.ones(self.nchoices)
        self.thr = np.ones(self.nchoices) * 3
        self.beta_counters = np.zeros((3, self.nchoices))
        self.force_fit = False
        self.force_counters = False
        self.epsilon = epsilon
        self.algos = [0] * self.nchoices
        self.njobs = 1
        
    def input_trans(self, X):
        return OneHotEncoder(categories=[np.arange(self.nchoices)]).fit_transform(X)
    
    def fit(self, X, a, r, warm_start=False):
        if not isinstance(self.base, Neural):
            X = self.input_trans(X)
        Parallel(n_jobs=self.njobs, verbose=0, require="sharedmem")\
                    (delayed(self._full_fit_single)\
                            (choice, X, a, r) for choice in range(self.nchoices))
        
        report = Report()
        self.arm_eval(report, X, a, r)
        self.report_list.append(report)
        
    def _full_fit_single(self, choice, X, a, r):
        st = time.time()
        this_choice = (a == choice)
        yclass = r[this_choice]
        n_pos = (yclass > 0.).sum()
        self.algos[choice] = deepcopy(self.base)
        copy_time = time.time() - st

#         if (n_pos < self.thr[choice]) or ((yclass.shape[0] - n_pos) < self.thr[choice]):
#             if not self.force_fit:
#                 self.algos[choice] = _BetaPredictor(self.alpha[choice] + n_pos,
#                                                     self.beta[choice] + yclass.shape[0] - n_pos,
#                                                     choice)
#                 return
        if n_pos == 0:
            if not self.force_fit:
                self.algos[choice] = _ZeroPredictor()
                return
        if n_pos == yclass.shape[0]:
            if not self.force_fit:
                self.algos[choice] = _OnePredictor()
                return

        arms_to_update = None
        if (arms_to_update is None) or (choice in arms_to_update):
            xclass = X[this_choice, :]
            self.algos[choice].fit(xclass, yclass)
        
    def _fit(self, X, a, r, warm_start=False):
        self.algos = [deepcopy(self.base) for i in range(self.nchoices)]
        for choice in range(self.nchoices):
            self._full_fit_single(choice, X, a, r)
                
    def _update_beta_counters(self, yclass, choice):
        if (self.beta_counters[0, choice] == 0) or (self.force_counters):
            n_pos = (yclass > 0.).sum()
            self.beta_counters[1, choice] += n_pos
            self.beta_counters[2, choice] += yclass.shape[0] - n_pos
            if (self.beta_counters[1, choice] > self.thr[choice]) and (self.beta_counters[2, choice] > self.thr[choice]):
                self.beta_counters[0, choice] = 1
        
    def predict_proba(self, X):
        if not isinstance(self.base, Neural):
            X = self.input_trans(X)
        preds = np.zeros((X.shape[0], self.nchoices))
        for choice in range(self.nchoices):
            preds[:, choice] = self.algos[choice].predict_proba(X)[:, 1]
        return preds
    
    def predict(self, local_X):
        n = local_X.shape[0]
        preds = self.predict_proba(local_X)
        self.evaluate(self.report_list[-1], preds, local_X)
        preds = preds[:n]
        pred = np.argmax(preds, axis=1).astype(int)
        for e, i, j in zip(range(n), np.random.random(n), np.random.randint(0, self.nchoices, n)):
            if i < self.epsilon:
                pred[e] = j
        return pred
import torch
import pickle
import logging
import numpy as np


class _FixedPredictor:
    def __init__(self):
        pass

    def fit(self, X=None, y=None, sample_weight=None):
        pass

    def decision_function_w_sigmoid(self, X):
        return self.decision_function(X)


class _ZeroPredictor(_FixedPredictor):

    def predict_proba(self, X):
        return np.c_[np.ones((X.shape[0], 1)),  np.zeros((X.shape[0], 1))]

    def decision_function(self, X):
        return np.zeros(X.shape[0])

    def predict(self, X):
        return np.zeros(X.shape[0])


class _OnePredictor(_FixedPredictor):

    def predict_proba(self, X):
        return np.c_[np.zeros((X.shape[0], 1)),  np.ones((X.shape[0], 1))]

    def decision_function(self, X):
        return np.ones(X.shape[0])

    def predict(self, X):
        return np.ones(X.shape[0])


class _RandomPredictor(_FixedPredictor):
    def __init__(self, random_state):
        self.random_state = _check_random_state(random_state)

    def _gen_random(self, X):
        return self.random_state.random(size = X.shape[0])

    def predict(self, X):
        return (self._gen_random(X) >= .5).astype('uint8')

    def decision_function(self, X):
        return self._gen_random(X)

    def predict_proba(self, X):
        pred = self._gen_random(X)
        return np.c[pred, 1. - pred]


class _BetaPredictor(_FixedPredictor):
    def __init__(self, a, b, random_state):
        self.a = a
        self.b = b
        self.random_state = _check_random_state(random_state)

    def predict_proba(self, X):
        preds = self.random_state.beta(self.a, self.b, size = X.shape[0]).reshape((-1, 1))
        return np.c_[1.0 - preds, preds]

    def decision_function(self, X):
        return self.random_state.beta(self.a, self.b, size = X.shape[0])

    def predict(self, X):
        return (self.random_state.beta(self.a, self.b, size = X.shape[0])).astype('uint8')

    def exploit(self, X):
        return np.repeat(self.a / self.b, X.shape[0])


def _check_random_state(random_state):
    if random_state is None:
        return np.random.Generator(np.random.MT19937())
    if isinstance(random_state, np.random.Generator):
        return random_state
    elif isinstance(random_state, np.random.RandomState) or (random_state == np.random):
        random_state = int(random_state.randint(np.iinfo(np.int32).max) + 1)
    if isinstance(random_state, float):
        random_state = int(random_state)
    #assert random_state > 0
    return np.random.Generator(np.random.MT19937(seed = random_state))


class Base():
    
    def __init__(self, nchoices, name=None, seed=87):
        self.nchoices = nchoices
        self.seed = seed
        self.name = name
        # report
        self.local_loss = []
        self.report_list = []
        
    def plot_arm_loss(self):
        hist = defaultdict(list)
        for report in self.report_list:
            for arm, info in report.arm_loss.items():
                hist[arm].append(info['loss'])

        for arm in hist:
            plt.plot([i for i in self.history[arm]], label=str(arm))
        plt.legend()
        
    def plot_local_loss(self):
        plt.plot([report.local_loss for report in self.report_list])
        
    def plot_global_loss(self):
        plt.plot([report.global_loss for report in self.report_list])
        
    def plot_gt_pred(self):
        plt.plot([i.gt_pred_error for i in self.report_list])
        
    def arm_eval(self, report, X, a, r):
        for choice in range(self.nchoices):
            this_choice = (a == choice)
            yclass = r[this_choice]
            xclass = X[this_choice, :]
            try:
                pred = self.algos[choice].predict_proba(xclass)[:, choice]
                loss = mean_squared_error(yclass, pred)
                self.arm_loss[choice].append({'loss': loss, 'num': len(yclass)})
                report.arm_loss[choice] = {'loss': loss, 'num': len(yclass)}
            except:
                pass
            
    def evaluate(self, report, pred, local_X, back=None):
        report.evaluate_pred(pred, local_X.shape[0])
        if back:
            report.evaluate_back(back, local_X.shape[0])
                    
    def save(self, it=None):
        path = f"checkpoint/rec/{self.name}{it}.pkl"
        logging.info(f"Save model to {path}.")
        if hasattr(self, "model"):
            if isinstance(self.model, torch.nn.Module):
                torch.save(self.model, path)
            else:
                with open(path, "wb") as f:
                    pickle.dump(self.model, f)
        elif hasattr(self, "algos"):
            with open(path, "wb") as f:
                pickle.dump(self.algos, f)
        else:
            raise Exception("fucl")
            
    def load(self, it=None):
        path = f"checkpoint/rec/{self.name}{it}.pkl"
        logging.info(f"Load model from {path}.")
        if hasattr(self, "model"):
            if isinstance(self.model, torch.nn.Module):
                self.model = torch.load(path)
            else:
                with open(path, "rb") as f:
                    self.model = pickle.load(f)
        elif hasattr(self, "algos"):
            with open(path, "rb") as f:
                self.algos = pickle.load(f)
        else:
            raise Exception("fuck")
            
    def fit(self):
        raise NotImplementedError
        
    def predict(self):
        raise NotImplementedError
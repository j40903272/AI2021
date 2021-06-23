import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, LBFGS
from torch.optim.lr_scheduler import MultiplicativeLR

from .base import Base
from .neural import NN
from ..utils import set_seed


class Value(Base):
    
    def __init__(self, nchoices, epsilon=0.0, ts=False, seed=87, name=None):
        
        super().__init__(nchoices, name, seed)
        
        self.model = NN(16, 1, 8, 3, 0.1, emb_dim=8).cuda()
        self.optim = Adam(self.model.parameters(), lr=1e-2, weight_decay=1e-5)
        self.scheduler = MultiplicativeLR(self.optim, lr_lambda=lambda epoch: 0.99)
        self.epsilon = epsilon
        self.ts = ts
        self.history = defaultdict(list)
        set_seed(seed)
        
    def fit(self, X, a, r, warm_start=False, verbose=False):
        #set_seed(self.seed)
        Y = np.zeros(len(r))
        for e, (i, j) in enumerate(zip(a, r)):
            Y[e] = j
        
        AA = torch.from_numpy(a).long().cuda().reshape(-1, 1)
        XX = torch.from_numpy(X).long().cuda()
        YY = torch.from_numpy(Y).float().cuda()
        scale = 1 # int(len(XX) / 500) + 1
        self.history['scale'].append(scale)
        for param_group in self.optim.param_groups:
            param_group['lr'] = 1e-2 * scale
        
        for e in range(10):
            self.model.train()
            running_loss = []
            for i in range(50*scale):
                logit = self.model(XX, AA).squeeze()
                loss = F.binary_cross_entropy_with_logits(logit, YY)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                running_loss.append(loss.item())
            #scheduler.step()

            self.model.eval()
            logit = self.model(XX, AA).squeeze()
            loss = F.binary_cross_entropy_with_logits(logit, YY)
            
            train_loss = np.mean(running_loss)
            eval_loss = loss.item()
            if verbose:
                print(train_loss, eval_loss)
            self.history['train_loss'].append(train_loss)
            self.history['eval_loss'].append(eval_loss)
            
        report = Report()
        self.arm_eval(report, X, a, r)
        self.report_list.append(report)
        
    def arm_eval(self, report, X, a, r):
        return
        
    def predict_proba(self, X):
        all_action = torch.from_numpy(np.arange(9724)).long().cuda()
        XX = torch.from_numpy(X).long().cuda()
        proba = []
        for x in XX:
            x = torch.stack([x]*self.nchoices)
            proba.append(F.sigmoid(self.model(x, all_action)).detach().cpu().numpy().flatten().tolist())
        return np.array(proba)
        
    def predict(self, local_X):
        self.model.eval()
        preds = self.predict_proba(local_X)
        self.evaluate(self.report_list[-1], preds, local_X)
        if self.ts:
            self.model.train()
            preds = self.predict_proba(X)
        n = local_X.shape[0]
        preds = preds[:n]
        pred = np.argmax(preds, axis=1).astype(int)
        for e, i, j in zip(range(n), np.random.random(n), np.random.randint(0, self.nchoices, n)):
            if i < self.epsilon:
                pred[e] = j
        return pred
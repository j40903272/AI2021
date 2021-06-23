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


class Policy(Base):
    
    def __init__(self, nchoices, name=None, ts=False, epsilon=0.0, seed=87):
        
        super().__init__(nchoices, name, seed)
        
        self.model = NN(8, nchoices, 8, 3, 0.1).cuda()
        self.optim = Adam(self.model.parameters(), lr=5e-2, weight_decay=1e-7)
        #self.scheduler = MultiplicativeLR(self.optim, lr_lambda=lambda epoch: 0.99)
        self.history = defaultdict(list)
        self.epsilon = epsilon
        self.ts = ts
        set_seed(self.seed)
        
    def fit(self, X, a, r, warm_start=False, verbose=False):
        #set_seed(self.seed)
        mask = np.zeros((len(a), self.nchoices))
        Y = np.zeros((len(r), self.nchoices))
        for e, (i, j) in enumerate(zip(a, r)):
            mask[e][i] = 1
            Y[e][i] = j
        
        MM = torch.from_numpy(mask).float().cuda()
        XX = torch.from_numpy(X).long().cuda()
        YY = torch.from_numpy(Y).float().cuda()
        scale = 1 #int(len(XX) / 200)
        
        for e in range(10):
            self.model.train()
            running_loss = []
            for i in range(100*scale):
                logit = self.model(XX) * MM
                # loss = F.mse_loss(logit, YY)
                loss = F.binary_cross_entropy_with_logits(logit, YY)
                self.optim.zero_grad()
                loss.backward()
                # print(self.model.fc_list[-1].weight.grad.sum())
                self.optim.step()
                running_loss.append(loss.item())
            #scheduler.step()

            self.model.eval()
            logit = self.model(XX) * MM
            # loss = F.mse_loss(logit, YY)
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
        
    def predict_proba(self, X):
        XX = torch.from_numpy(X).long().cuda()
        return F.sigmoid(self.model(XX)).detach().cpu().numpy()
        
    def predict(self, local_X):
        self.model.eval()
        preds = self.predict_proba(X)
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
    
    def plot_history(self):
        plt.plot(models[3].history['train_loss'])

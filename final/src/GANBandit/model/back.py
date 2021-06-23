import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, LBFGS
from torch.optim.lr_scheduler import MultiplicativeLR

from GANBandit.model.base import Base
from GANBandit.model.value import Value
from GANBandit.model.neural import NN
from GANBandit.utils import set_seed


class BackValue(Value):
    
    def __init__(self, nchoices, epsilon=0.0, ts=False, seed=87, name=None):
        
        super().__init__(nchoices, epsilon, ts, seed, name)
        self.back_logs = []
        self.checkpoints = []
        
    def create_tree(self):
        self.item_emb = self.model.item_emb.weight.detach().cpu().numpy().copy()[:-1]
        self.tree = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
        self.tree.fit(self.item_emb)
    
    def predict(self, local_X):
        self.create_tree()
        preds = self.predict_proba(local_X)
        back, logs = self.parallel_back(local_X, preds)
        self.back_logs.append(logs)
        self.evaluate(self.report_list[-1], preds, local_X, back)
        
        n = local_X.shape[0]
        back = back[:n]
        for e, i, j in zip(range(n), np.random.random(n), np.random.randint(0, self.nchoices, n)):
            if i < self.epsilon:
                back[e] = j
                
#         if len(self.back_logs) % 10 == 0:
#             checkpoint = deepcopy(self.model)
#             self.checkpoints.append(checkpoint.cpu())

        return np.array(back)
    
    def back(self, X, proba):
        if self.ts:
            self.model.train()
        XX = torch.from_numpy(X).long().cuda()
        preds = []
        logs = {
            'is_real_best': [],
            'real_best_score': [],
            'ours_real_score': [],
            'loss_list': []
        }
        
        for x, p in tqdm(zip(XX, proba)):
            x = x.view(1, -1)
            arm, is_real_best, real_best_score, ours_real_score, loss_list = self.gradient_ascend(x, p)
            preds.append(arm)
            
            logs['is_real_best'].append(is_real_best)
            logs['real_best_score'].append(real_best_score)
            logs['ours_real_score'].append(ours_real_score)
            logs['loss_list'].append(loss_list)
        
        return preds, logs

    def gradient_ascend(self, XXX, proba):
        set_seed(0)
        best = None
        best_score = -10000
        loss_list = []
        
        real_best = np.argmax(proba)
        real_best_score = proba[real_best]

        for multistart in range(1):
            action = torch.full((1, self.nchoices), 1/self.nchoices, requires_grad=True, device="cuda")
            context = torch.tensor(XXX, requires_grad=True, device="cuda")
            optim = Adam([action], lr=1e-3, weight_decay=1e-6)

            for i in range(100):
                inp = torch.cat([context, action.clone()], axis=1)
                x = inp
                for e, fc in enumerate(self.model.fc_list):
                    x = fc(x)
                    if e < self.model.layers-1:
                        x = self.model.drop(x)
                        x = F.leaky_relu(x)

                # backward
                # regu = torch.sum(torch.square(inp - torch.mean(inp))) * 1e-1
                logit = x
                x = F.sigmoid(x)
                loss = -x
                #loss = F.binary_cross_entropy_with_logits(logit, torch.ones(x.shape).float().cuda())
                optim.zero_grad()
                loss.backward()
                optim.step()

                w = inp.data
                w = w.clamp(0, 1)
                inp.data = w
                loss_list.append(loss.item())

            
            idx = np.argmax(inp.detach().cpu().numpy()[0, 100:])
            ours_real_score = proba[idx]
            if ours_real_score > best_score:
                best_score = ours_real_score
                best = idx
        
        is_real_best = (real_best==best)
        return best, is_real_best, real_best_score, best_score, loss_list
    
    
    def parallel_back(self, X, proba):
        if self.ts:
            self.model.train()
        preds = []
        logs = defaultdict(list)

        results = Parallel(n_jobs=8, verbose=0)(delayed(self.parallel_gradient_ascend)(x, p) for x, p in zip(X, proba))

#         results = []
#         for x, p in tqdm(zip(X, proba), total=len(proba)):
#             results.append(self.parallel_gradient_ascend(x, p))
        
        for arm, is_real_best, real_best_score, ours_real_score, loss_list, dist in results:
            preds.append(arm)
            logs['is_real_best'].append(is_real_best)
            logs['real_best_score'].append(real_best_score)
            logs['ours_real_score'].append(ours_real_score)
            logs['loss_list'].append(loss_list)
            logs['dist'].append(dist)
        
        return preds, logs

    def parallel_gradient_ascend(self, XXX, proba):
        set_seed(0)
        XXX = torch.from_numpy(XXX).long().cuda().view(1, -1)
        XXX = self.model.user_emb(XXX).squeeze(1)
        best = None
        best_score = -10000
        best_dist = 0
        loss_list = []
        
        real_best = np.argmax(proba)
        real_best_score = proba[real_best]

        for multistart in range(10):
            # action = torch.full((1, self.nchoices), 1/self.nchoices, requires_grad=True, device="cuda")
            action = torch.rand(1, 8, requires_grad=True, device="cuda")
            if True:
                best_emb = self.item_emb[real_best].reshape(1, -1)
                action = torch.from_numpy(best_emb).float()
                action = torch.tensor(action, requires_grad=True, device="cuda")
            
            context = torch.tensor(XXX, requires_grad=True, device="cuda")
            optim = Adam([action], lr=1e-2, weight_decay=1e-6)

            for i in range(30):
                inp = torch.cat([context, action.clone()], axis=1)
                x = inp
                for e, fc in enumerate(self.model.fc_list):
                    x = fc(x)
                    if e < self.model.layers-1:
                        x = self.model.drop(x)
                        x = F.leaky_relu(x)

                # backward
                # regu = torch.sum(torch.square(inp - torch.mean(inp))) * 1e-1
                logit = x
                x = F.sigmoid(x)
                loss = -x
                #loss = F.binary_cross_entropy_with_logits(logit, torch.ones(x.shape).float().cuda())
                optim.zero_grad()
                loss.backward()
                optim.step()

#                 w = inp.data
#                 w = w.clamp(0, 1)
#                 inp.data = w
                loss_list.append(loss.item())
    
                if x.item() > 0.7:
                    break

            dist, idx = self.tree.kneighbors(inp.detach().cpu().numpy()[:1, 8:])
            dist, idx = dist[0][0], idx[0][0]
            ours_real_score = proba[idx]
            if ours_real_score > best_score:
                best_score = ours_real_score
                best = idx
                best_dist = dist
        
        is_real_best = (real_best==best)
        return best, is_real_best, real_best_score, best_score, loss_list, best_dist
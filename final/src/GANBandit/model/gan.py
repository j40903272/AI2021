import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, LBFGS
from torch.optim.lr_scheduler import MultiplicativeLR

from .base import Base
from ..utils import set_seed


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * 0.1


def grad_reverse(x):
    return GradReverse.apply(x)


class Discriminator(torch.nn.Module):

    def __init__(self, inp_dim, out_dim, hidden, layers, drop, emb_dim=0):
        # inp_dim = emb_dim * 2
        super().__init__()
        self.layers = layers
        self.fc_list = nn.ModuleList([torch.nn.Linear(hidden, hidden) for _ in range(layers)])
        self.fc_list[-1] = torch.nn.Linear(hidden, out_dim)
        self.fc_list[0] = torch.nn.Linear(inp_dim, hidden)
        self.drop = torch.nn.Dropout(drop)
        self.emb_dim = emb_dim
        if self.emb_dim:
            self.item_emb = torch.nn.Embedding(9724+1, emb_dim)
            self.user_emb = torch.nn.Embedding(610+1, emb_dim)
        else:
            self.user_emb = torch.nn.Embedding(610+1, hidden)

    def forward(self, user, item=None):
        if self.emb_dim:
            item = self.item_emb(item).squeeze(1)
            user = self.user_emb(user).squeeze(1)
            x = torch.cat([user, item], dim=1)
        else:
            x = self.user_emb(user).squeeze(1)
        
        return self.emb_forward(x)
    
    def emb_forward(self, x):
        for e, fc in enumerate(self.fc_list):
            x = fc(x)
            if e < self.layers-1:
                x = self.drop(x)
                x = F.leaky_relu(x)
        return x


class Generator(torch.nn.Module):

    def __init__(self, inp_dim, out_dim, hidden, layers, drop, emb_dim=0):
        super().__init__()
        self.layers = layers
        self.fc_list = nn.ModuleList([torch.nn.Linear(hidden, hidden) for _ in range(layers)])
        self.fc_list[-1] = torch.nn.Linear(hidden, out_dim)
        self.fc_list[0] = torch.nn.Linear(inp_dim, hidden)
        self.drop = torch.nn.Dropout(drop)

    def forward(self, x, z):
        x = torch.cat([x, z], dim=1)
        for e, fc in enumerate(self.fc_list):
            x = fc(x)
            if e < self.layers-1:
                x = self.drop(x)
                x = F.leaky_relu(x)
        return x


class EmbPredictor(nn.Module):
    def __init__(self, inp_dim, out_dim, hidden, layers, drop):
        super(EmbPredictor, self).__init__()
        self.fc_list = nn.ModuleList([torch.nn.Linear(hidden, hidden) for _ in range(layers)])
        self.fc_list[-1] = torch.nn.Linear(hidden, out_dim)
        self.fc_list[0] = torch.nn.Linear(inp_dim, hidden)
        self.drop = torch.nn.Dropout(drop)
        self.layers = layers

    def forward(self, x):
        x = grad_reverse(x)
        for e, fc in enumerate(self.fc_list):
            x = fc(x)
            if e < self.layers-1:
                x = self.drop(x)
                x = F.leaky_relu(x)
        return x


class GAN():
    
    def __init__(self, generator, discriminator, embpredictor, regu=0):
        self.G = generator
        self.D = discriminator
        self.P = embpredictor
        self.Gopt = Adam(self.G.parameters(), lr=1e-3, weight_decay=1e-5)
        self.Dopt = Adam(self.D.parameters(), lr=1e-2, weight_decay=1e-5)
        self.Popt = Adam(self.P.parameters(), lr=1e-2, weight_decay=1e-5)
        #self.scheduler = MultiplicativeLR(self.optim, lr_lambda=lambda epoch: 0.99)
        self.history = defaultdict(list)
        self.verbose = False
        self.regu = regu
        
    def eval(self):
        self.D.eval()
        self.G.eval()
        
    def train(self):
        self.D.train()
        self.G.train()
    
    def create_tree(self):
        self.item_emb = self.D.item_emb.weight.detach().cpu().numpy().copy()[:-1]
        self.tree = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
        self.tree.fit(self.item_emb)
        
    def get_nearest(self, inp, n_neighbors=1):
        dist, idx = self.tree.kneighbors(inp.detach().cpu().numpy(), n_neighbors=n_neighbors)
        return dist, idx
        
    def fit(self, XX, AA, YY):
        
        self.last_fit = (XX, AA, YY)
        
        for e in range(10):
            
            self.G.train()
            self.D.train()
            running_loss_D = []
            running_loss_G = []
            running_loss_P = []
            running_loss_PG = []
            for i in range(300*scale):
                
                
                # train discrimiator
                for _ in range(1):
                    logit = self.D(XX, AA).squeeze()
                    loss = F.binary_cross_entropy_with_logits(logit, YY)
                    self.Dopt.zero_grad()
                    loss.backward()
                    self.Dopt.step()
                    running_loss_D.append(loss.item())
                    
                # random zeros
                for _ in range(0):
                    RandA = torch.from_numpy(np.random.randint(0, n_class, len(AA))).cuda()         
                    logit = self.D(XX, RandA).squeeze()
                    loss = F.binary_cross_entropy_with_logits(logit, torch.zeros((len(YY))).cuda())
                    self.Dopt.zero_grad()
                    loss.backward()
                    self.Dopt.step()
                    running_loss_D.append(loss.item())

                # train generator
                for _ in range(3):
                    #R = torch.randn((AA.shape[0], 8)).cuda()
                    R = torch.zeros((AA.shape[0], 8)).cuda()
                    user_emb = self.D.user_emb(XX).squeeze(1)
                    argmax_item_emb = self.G(user_emb, R)
                    inp = torch.cat([user_emb, argmax_item_emb], dim=1)
                    logit = self.D.emb_forward(inp).squeeze()
                    #loss = F.mse_loss(F.sigmoid(logit), torch.ones(logit.shape).cuda())
                    loss = F.binary_cross_entropy_with_logits(logit, torch.ones(logit.shape).cuda())
                    self.Gopt.zero_grad()
                    loss.backward()
                    self.Gopt.step()
                    running_loss_G.append(loss.item())

                # regularization
                for _ in range(self.regu):
                    # train emb predictor
                    for _ in range(1):
                        Rand = torch.from_numpy(np.random.randint(0, n_class, len(argmax_item_emb))).cuda()
                        real_emb = self.D.item_emb(Rand).squeeze(1)
                        R = torch.zeros((AA.shape[0], 8)).cuda()
                        user_emb = self.D.user_emb(XX).squeeze(1)
                        fake_emb = self.G(user_emb, R)
                        real_label = torch.ones(len(fake_emb)).cuda()
                        fake_label = torch.zeros(len(fake_emb)).cuda()
                        embs = torch.cat([real_emb, fake_emb], dim=0)
                        labels = torch.cat([real_label, fake_label], dim=0).unsqueeze(1)
                        logit = self.P(embs)
                        loss = F.binary_cross_entropy_with_logits(logit, labels)
                        self.Popt.zero_grad()
                        loss.backward()
                        self.Popt.step()
                        running_loss_P.append(loss.item())

                    # reverse generator
                    for _ in range(1):
                        Rand = torch.from_numpy(np.random.randint(0, n_class, len(argmax_item_emb))).cuda()
                        real_emb = self.D.item_emb(Rand).squeeze(1)
                        R = torch.zeros((AA.shape[0], 8)).cuda()
                        user_emb = self.D.user_emb(XX).squeeze(1)
                        fake_emb = self.G(user_emb, R)
                        real_label = torch.ones(len(fake_emb)).cuda()
                        fake_label = torch.zeros(len(fake_emb)).cuda()
                        embs = torch.cat([real_emb, fake_emb], dim=0)
                        labels = torch.cat([real_label, fake_label], dim=0).unsqueeze(1)
                        ####
                        embs = grad_reverse(embs)
                        logit = self.P(embs)
                        loss = F.binary_cross_entropy_with_logits(logit, labels)
                        self.Gopt.zero_grad()
                        loss.backward()
                        self.Gopt.step()
                        running_loss_PG.append(loss.item())
                        
                    # dist regularization
                    for _ in range(0):
                        Rand = torch.from_numpy(np.random.randint(0, n_class, len(argmax_item_emb))).cuda()
                        real_emb = self.D.item_emb(Rand).squeeze(1)
                        R = torch.zeros((AA.shape[0], 8)).cuda()
                        user_emb = self.D.user_emb(XX).squeeze(1)
                        fake_emb = self.G(user_emb, R)
                        
                        loss = F.mse_loss(fake_emb, real_emb) * 0.001
                        #loss = F.cosine_embedding_loss(real_emb, fake_emb, -torch.ones(len(fake_emb)).cuda())
                        self.Gopt.zero_grad()
                        loss.backward()
                        self.Gopt.step()
                        running_loss_PG.append(loss.item())
                        
            
            
            d_train_loss = np.mean(running_loss_D)
            d_eval_loss = d_train_los
            g_train_loss = np.mean(running_loss_G)
            g_eval_loss = g_train_loss
            p_train_loss = np.mean(running_loss_P)
            pg_train_loss = np.mean(running_loss_PG)
                
            self.history['g_train_loss'].append(g_train_loss)
            self.history['g_eval_loss'].append(g_eval_loss)
            self.history['d_train_loss'].append(d_train_loss)
            self.history['d_eval_loss'].append(d_eval_loss)
            self.history['p_train_loss'].append(p_train_loss)
            self.history['pg_train_loss'].append(pg_train_loss)
            
        self.create_tree()
            
    def get_D_argmax(self, XX, return_value=True):
        with torch.no_grad():
            XX = torch.from_numpy(XX).long().cuda()
            all_action = torch.from_numpy(np.arange(9724)).long().cuda()
            argmax_list = []
            value_list = []
            for x in XX:
                x = torch.stack([x]*len(all_action))
                logit = self.D(x, all_action).squeeze(1)
                value = F.sigmoid(logit)
                argmax = torch.argmax(logit, dim=0)
                value_list.append(value)
                argmax_list.append(argmax)
            
            if return_value:
                return torch.stack(argmax_list).detach().cpu().numpy().flatten(), torch.stack(value_list).detach().cpu().numpy()
            else:
                return torch.stack(argmax_list).detach().cpu().numpy().flatten()
    
    def get_G_argmax(self, XX, return_value=True, multitrial=1, topk=1, ts=False):
        with torch.no_grad():
            x = torch.from_numpy(XX).cuda().long()
            user_emb = self.D.user_emb(x).squeeze(1)
            best = torch.zeros(len(XX), 1).long().cuda()
            best_dvalue = torch.zeros(len(XX), 1).cuda()
            best_gvalue = torch.zeros(len(XX), 1).cuda()
            
            for _ in range(multitrial):
                if False:
                    R = torch.randn((len(XX), 8)).cuda()
                else:
                    R = torch.zeros((len(XX), 8)).cuda()
                
                out_emb = self.G(user_emb, R)
                dist, argmaxs = self.get_nearest(out_emb, topk)

                for i in range(topk):
                    argmax = argmaxs[:, i:i+1]
                    a = torch.from_numpy(argmax).long().cuda()
                    Dvalue = F.sigmoid(self.D(x, a))
                    inp = torch.cat([user_emb, out_emb], dim=1)
                    Gvalue = F.sigmoid(self.D.emb_forward(inp))

                    best = torch.where(Dvalue > best_dvalue, a, best)
                    best_dvalue = torch.where(Dvalue > best_dvalue, Dvalue, best_dvalue)
                    best_gvalue = torch.where(Dvalue > best_dvalue, Gvalue, best_gvalue)
            
            argmax = best.detach().cpu().numpy().flatten()
            best_dvalue = best_dvalue.detach().cpu().numpy().flatten()
            best_gvalue = best_gvalue.detach().cpu().numpy().flatten()
            if return_value:
                return argmax, best_dvalue, best_gvalue, out_emb.detach().cpu().numpy()
            else:
                return argmax.flatten()
                

class Gan(Base):
    
    def __init__(self, nchoices, epsilon=0.0, ts=False, seed=87, name=None, regu=0):
        
        super().__init__(nchoices, name, seed)
        
        D = Discriminator(16, 1, 8, 3, 0.1, emb_dim=8).cuda()
        G = Generator(16, 8, 8, 3, 0.2, emb_dim=8).cuda()
        E = EmbPredictor(8, 1, 2, 4, 0.0).cuda()
        self.model = GAN(G, D, E, regu=regu)
        self.epsilon = epsilon
        self.ts = ts
        self.regu = 0
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
        
        self.model.fit(XX, AA, YY)
            
        report = Report()
        self.arm_eval(report, X, a, r)
        self.report_list.append(report)
        
    def arm_eval(self, report, X, a, r):
        return
        
    def predict_proba(self, X):
        pred, proba = self.model.get_D_argmax(X)
        return pred, proba
        
    def predict(self, local_X):
        self.model.eval()
        if self.ts:
            self.model.train()
        pred, proba = self.predict_proba(local_X)
        back, dvalue, gvalue, out_emb = self.model.get_G_argmax(local_X, multitrial=1, topk=10, ts=self.ts)
        back = back.astype(int).flatten()

        self.report_list[-1].predict = proba
        self.report_list[-1].local_X = local_X
        self.report_list[-1].back = back
        
        self.report_list[-1].dvalue = dvalue
        self.report_list[-1].gvalue = gvalue
        self.report_list[-1].out_emb = out_emb
        
        
        n = len(local_X)
        pred = back[:n]
#         pred = np.argmax(proba, axis=1).astype(int)
        for e, i, j in zip(range(n), np.random.random(n), np.random.randint(0, self.nchoices, n)):
            if i < self.epsilon:
                pred[e] = j
                
        return np.array(pred)
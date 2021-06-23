import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
# from torch.optim.lr_scheduler import MultiplicativeLR


class NN(torch.nn.Module):

    def __init__(self, inp_dim, out_dim, hidden, layers, drop, emb_dim=0):
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
            x = torch.cat([item, user], dim=1)
        else:
            x = self.user_emb(user).squeeze(1)
        
        for e, fc in enumerate(self.fc_list):
            x = fc(x)
            if e < self.layers-1:
                x = self.drop(x)
                x = F.leaky_relu(x)
        return x


class Neural():
    
    def __init__(self, inp_dim=8, out_dim=1):
        self.model = NN(inp_dim, out_dim, 8, 3, 0.1).cuda()
        self.optim = Adam(self.model.parameters(), lr=1e-2, weight_decay=1e-5)
        #self.scheduler = MultiplicativeLR(self.optim, lr_lambda=lambda epoch: 0.99)
        self.history = defaultdict(list)
        
    def fit(self, X, Y, verbose=False):
        XX = torch.from_numpy(X).long().cuda()
        YY = torch.from_numpy(Y).float().cuda().unsqueeze(1)
        scale = 1#int(len(XX) / 200)
        
        for e in range(1):
            self.model.train()
            running_loss = []
            for i in range(30*scale):
                logit = self.model(XX).view(-1, 1)
                loss = F.binary_cross_entropy_with_logits(logit, YY)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                running_loss.append(loss.item())
            #scheduler.step()

            self.model.eval()
            logit = self.model(XX)
            loss = F.binary_cross_entropy_with_logits(logit, YY)
            
            train_loss = np.mean(running_loss)
            eval_loss = loss.item()
            if verbose:
                print(train_loss, eval_loss)
            self.history['train_loss'].append(train_loss)
            self.history['eval_loss'].append(eval_loss)
        
    def predict_proba(self, X):
        XX = torch.from_numpy(X).long().cuda()
        proba = F.sigmoid(self.model(XX)).detach().cpu().numpy().flatten()
        return np.stack([1-proba, proba], axis=1)
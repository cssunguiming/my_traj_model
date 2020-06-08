import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class Position_Embedding(nn.Module):

    def __init__(self, d_model, max_len=800):
        super(Position_Embedding, self).__init__()

        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model) )

        pe[:,0::2] = torch.sin(position * div)
        pe[:,1::2] = torch.cos(position * div)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return x * math.sqrt(self.d_model)

class Token_Embedding(nn.Module):

    def __init__(self, token_size, d_model):
        super(Token_Embedding, self).__init__()

        self.d_model = d_model
        self.token_embed = nn.Embedding(token_size, d_model, padding_idx=0)
        # self.token_embed = nn.Embedding(token_size, d_model-10, padding_idx=0)
        self.time_embed = nn.Embedding(49, d_model, padding_idx=0)

    def forward(self, x, time):
        Embed = self.token_embed(x) + self.time_embed(time)
        # Embed = self.token_embed(x)

        # Embed = torch.cat((self.token_embed(x), self.time_embed(time)), dim=-1)
        return Embed * math.sqrt(self.d_model)

class Traj_Embedding(nn.Module):

    def __init__(self, d_model):
        super(Traj_Embedding, self).__init__()

        self.d_model = d_model
        self.traj_embedding = nn.Embedding(3, d_model, padding_idx=0)

    def forward(self, x):
        return self.traj_embedding(x) * math.sqrt(self.d_model)

class Bert_Embedding(nn.Module):

    def __init__(self, token_size, d_model, dropout):
        super(Bert_Embedding, self).__init__()

        self.token_embed = Token_Embedding(token_size=token_size, d_model=d_model)
        self.posi_embed = Position_Embedding(d_model=d_model)
        # self.traj_embed = Traj_Embedding(d_model=d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time):
        # x = self.dropout(self.token_embed(x, time) + self.posi_embed(x) + self.traj_embed(seg_label))
        x = self.dropout(self.token_embed(x, time) + self.posi_embed(x))
        return x


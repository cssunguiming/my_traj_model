import torch
import torch.nn as nn
import torch.nn.functional as F
from bert_traj_model import Bert_Traj_Model 


class NextSentencePredict(nn.Module):

    def __init__(self, d_model):
        super(NextSentencePredict, self).__init__()

        self.Linear = nn.Linear(d_model, 2)

    def forward(self, x):
        
        x = self.Linear(x[:, 0])
        return F.softmax(x, dim=-1)


class Masked_LM(nn.Module):

    def __init__(self, token_size, d_model):
        super().__init__()

        self.Linear = nn.Linear(d_model, token_size)

    def forward(self, x):
        x = self.Linear(x)
        return F.softmax(x, dim=-1)


class Predict_Model(nn.Module):

    def __init__(self, Bert_Traj_Model, token_size, head_n=12, d_model=768, N_layers=12, dropout=0.1):
        super(Predict_Model, self).__init__()

        self.bert = Bert_Traj_Model

        self.place_Linear = nn.Linear(d_model, token_size)
        self.time_Linear = nn.Linear(d_model, 49)

        self.place_Linear.weight = Bert_Traj_Model.Embed.token_embed.token_embed.weight
        self.time_Linear.weight = Bert_Traj_Model.Embed.token_embed.time_embed.weight

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, input_time, max_len):
        x = self.bert(x, input_time, max_len)
        # logit: [batch_size, seq_size, d_]
        logit1 = self.place_Linear(x)
        logit2 = self.time_Linear(x)

        logit1 = logit1[:,max_len:]
        logit2 = logit2[:,max_len:]

        return logit1.contiguous().view(-1, logit1.size(-1)), logit2.contiguous().view(-1, logit2.size(-1))




import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Attn(nn.Module):
    
    def __init__(self, dropout=0.1):
        super(Attn, self).__init__()

        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        # Q: [batch_size, head_n, seq_size, d_q]
        d_k = Q.size(-1)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attn.masked_fill_(mask==0, -1e9)

        softmax = F.softmax(attn, dim=-1)
        
        if self.dropout is not None:
            softmax = self.dropout(softmax)

        output = torch.matmul(softmax, V)

        return output

class Mul_Attn(nn.Module):

    def __init__(self, head_n, d_model, d_q, d_k, d_v, dropout=0.1):
        super(Mul_Attn, self).__init__()

        self.d_model, self.head_n, self.d_q, self.d_k, self.d_v = d_model, head_n, d_q, d_k, d_v

        self.Linear_Q = nn.Linear(d_model, head_n*d_q)
        self.Linear_K = nn.Linear(d_model, head_n*d_k)
        self.Linear_V = nn.Linear(d_model, head_n*d_v)

        self.attn = Attn()
        self.Linear = nn.Linear(head_n*d_v, d_model)

    def forward(self, q, k, v, mask=None):
        # x: [batch_size, seq_size, d_model]
        batch_size = q.size(0)

        Q, K, V = (self.Linear_Q(q)/math.sqrt(self.d_model)).view(batch_size, -1, self.head_n, self.d_q).transpose(1,2), \
                  (self.Linear_K(k)/math.sqrt(self.d_model)).view(batch_size, -1, self.head_n, self.d_k).transpose(1,2), \
                  (self.Linear_V(v)/math.sqrt(self.d_model)).view(batch_size, -1, self.head_n, self.d_v).transpose(1,2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        attn_output = self.attn(Q, K, V, mask=mask)
        
        output = self.Linear(attn_output.transpose(1,2).contiguous().view(batch_size, -1, self.head_n*self.d_v))

        return output
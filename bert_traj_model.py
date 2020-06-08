import math
import numpy as np
import torch
import torch.nn as nn
import copy
from torch.autograd import Variable
import torch.nn.functional as F

from bert_traj_embedding import Bert_Embedding
from bert_traj_attn import Mul_Attn
from bert_traj_sublayer import PositionwiseFeedForward, SublayerConnection, LayerNorm


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_std_mask(tgt, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(
        subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask


class transformer_layer(nn.Module):

    def __init__(self, size, head_n, d_model, d_ff, dropout=0.1):
        super(transformer_layer, self).__init__()

        self.attn =  Mul_Attn(head_n=head_n, d_model=d_model, d_q=int(d_model/head_n), d_k=int(d_model/head_n), d_v=int(d_model/head_n), dropout=0.1)
        self.feed_forward =  PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.connect1 = SublayerConnection(size=size, dropout=dropout)
        self.connect2 = SublayerConnection(size=size, dropout=dropout)

    def forward(self, x, mask):

        self_attn = self.connect1(x, lambda x: self.attn(x, x, x, mask))
        output = self.connect2(self_attn, self.feed_forward)
        return output


# class Memory(nn.Module):

#     def __init__(self, user_size=4606, d_model=512):
#         super(Memory, self).__init__()

#         self.register_buffer('user_matrix', Variable(torch.Tensor(user_size, d_model)))
#         nn.init.normal_(self.user_matrix)

#     def reset(self):
#         nn.init.normal_(self.user_matrix)

#     def update(self, user, updates):
#         self.user_matrix[user] = updates

#     def read(self, user):
#         return self.user_matrix[user]


class Bert_Traj_Model(nn.Module):

    def __init__(self, token_size, user_size=1791, head_n=12, d_model=768, N_layers=12, dropout=0.1):
        super(Bert_Traj_Model, self).__init__()

        self.N_layers = N_layers
        self.Embed = Bert_Embedding(token_size=token_size, d_model=d_model, dropout=dropout)
        self.trans_layers = nn.ModuleList([copy.deepcopy(
            transformer_layer(size=d_model, head_n=head_n, d_model=d_model, d_ff=d_model*4, dropout=dropout)) for _ in range(N_layers)])
        self.layer_norm = LayerNorm(size=d_model)

    
    def forward(self, x, time, len_traj):
        # x:    [batch_size, seq_size] --> [batch_size, head_n, seq_size, seq_size]
        # mask: [batch_size, seq_size]
        len_s = x.size(-1)
        mask_pad = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1)
        mask_next = (1 - torch.triu(torch.ones((1, len_s, len_s), device=x.device), diagonal=1)).bool()
        # mask = mask_pad & mask_next

        mask_next[:,:, :len_traj] = True

        mask = mask_next & mask_pad 

        
        x = self.Embed(x, time)

        for i, layer in enumerate(self.trans_layers):
            x = layer(x, mask)

        return x

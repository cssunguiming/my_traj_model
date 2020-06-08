import torch
import torch.nn.functional as F
import numpy as np

def cal_loss_performance(logit1=None, logit2=None, label1=None, label2=None, Predict=False):

    if Predict:
        n = (label1.ne(0)).sum().item()

        loss_place = F.cross_entropy(logit1, label1, reduction='mean', ignore_index=0)
        n_cor = (label1.ne(0) * label1.eq(logit1.argmax(-1))).sum().item()

        loss_time = F.cross_entropy(logit2, label2, reduction='mean', ignore_index=0)
        n_time_cor = (label1.ne(0) * label2.eq(logit2.argmax(-1))).sum().item()

        # a1 = logit1.argmax(-1)
        # a2 = label1
        # length = len(a1)
        # b = []
        # for i in range(length):
        #     b.append((a1[i].item(), a2[i].item()))

        # print(b)
        # print("logit1:\n",logit1.argmax(-1))
        # print(''"label1:\n",label1)
        # print("n_cor:{}, n_total:{}".format(n_cor, n))
        # print("exit in cal loss")
        # exit()

        # loss = loss_place + loss_time
        loss = loss_place

        return loss, n, n_cor, n_time_cor
    else:
        loss_next = F.cross_entropy(logit1, label1, reduction='mean')
        loss_masked = F.cross_entropy(logit2, label2, reduction='mean', ignore_index=0)
        loss = loss_next + loss_masked

        n_next_sentence = len(label1)
        n_cor_next_sentence = logit1.argmax(-1).eq(label1).sum().item()
        n_masked_lm = (label2.ne(0)).sum().item()
        n_cor_masked_lm = (label2.ne(0) * label2.eq(logit2.argmax(-1))).sum().item()

        return loss, n_next_sentence, n_cor_next_sentence, n_masked_lm, n_cor_masked_lm

def get_acc(target, scores, eval_list):
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(10, 1)
    predx = idxx.cpu().numpy()
    acc = np.zeros((3, 1))
    ndcg = np.zeros((3, 1))
    for i, p in enumerate(predx):
        t = target[i]
        if t != 0:
            if t in p[:10] and t > 0: 
                acc[0] += 1
                rank_list = list(p[:10])
                rank_index = rank_list.index(t)
                # print("rand index",rank_index)
                # print("rand list",rank_list)
                ndcg[0] += 1.0 / np.log2(rank_index + 2)
            if t in p[:5] and t > 0:
                acc[1] += 1
                rank_list = list(p[:5])
                rank_index = rank_list.index(t)
                ndcg[1] += 1.0 / np.log2(rank_index + 2)
            if t == p[0] and t > 0:
                acc[2] += 1
                rank_list = list(p[:1])
                rank_index = rank_list.index(t)
                ndcg[2] += 1.0 / np.log2(rank_index + 2)
        else:
            pass

    eval_list[0] += acc[2]
    eval_list[1] += acc[1]
    eval_list[2] += acc[0]
    eval_list[3] += ndcg[2]
    eval_list[4] += ndcg[1]
    eval_list[5] += ndcg[0]

    return eval_list
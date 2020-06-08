import pickle
import numpy as np
import torch
from torch.autograd import Variable
from collections import deque

def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    # vec_size = len(vec)
    # pad_size = pad - vec_size

    # return np.concatenate((np.array(vec), np.zeros([pad_size], dtype=np.int32)), axis=dim)

    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)

    return torch.cat([vec, torch.zeros(*pad_size, dtype=torch.long)], dim=dim)



def generate_input_history(data_neural, mode, candidate=None):
    data = {}
    traj_idx = {}

    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        traj_id = data_neural[u][mode]

        data[u] = {} 

        for c, i in enumerate(traj_id):

            session = sessions[i]
            trace = {}
            # target = np.array([s[0] for s in session[1:]])

            target_loc_tim = [(0, 0)]
            target_loc_tim.extend([(s[0], s[1]) for s in session[1:]])
            target_loc_np = np.reshape(np.array([s[0] for s in target_loc_tim]), (-1))
            target_tim_np = np.reshape(np.array([s[1] for s in target_loc_tim]), (-1))

            # loc_tim = history
            loc_tim = [(0, 0)]
            loc_tim.extend([(s[0], s[1]) for s in session[:-1]])
            loc_np = np.reshape(np.array([s[0] for s in loc_tim]), (-1))
            tim_np = np.reshape(np.array([s[1] for s in loc_tim]), (-1))

            len_traj = np.array([len(loc_np)])
            # print(len_traj)

            trace['loc'] = Variable(torch.LongTensor(loc_np))
            trace['tim'] = Variable(torch.LongTensor(tim_np))
            trace['target_loc'] = Variable(torch.LongTensor(target_loc_np))
            trace['target_tim'] = Variable(torch.LongTensor(target_tim_np))
            trace['len_traj'] = Variable(torch.LongTensor(len_traj))

            data[u][i] = trace
        traj_idx[u] = traj_id
        
    # data[u][i]: {'history_loc', 'hitory_tim', 'history_count',
    #        'loc', 'tim', 'target'}    
    # traj_idx: {u: train_id or test_id}
    return data, traj_idx

def generate_queue(traj_idx, mode):

    user = list(traj_idx.keys())
    queue = deque()
    if mode=='random':
        initial_queue = {}
        for u in user:
            initial_queue[u] = deque(traj_idx[u])
        queue_left = 1
        while queue_left > 0:
            np.random.shuffle(user)
            for j, u in enumerate(user):
                if len(initial_queue[u])>0:
                    queue.append((u, initial_queue[u].popleft()))
                if j >= int(0.01 * len(user)):
                    break
            queue_left = sum([1 for x in initial_queue if len(initial_queue[x])>0])
    elif mode=='normal':
        for u in user:
            for i in traj_idx[u]:
                queue.append((u, i))

    return queue


def train_epoch(train_data, train_traj_idx, optimizer, device, epoch, batch_size=4):


    train_queue = generate_queue(train_traj_idx, 'random')
    len_queue = len(train_queue)
    len_batch = int(np.ceil((len_queue/batch_size)))
    train_queue = [[train_queue.popleft() for _ in range(min(batch_size, len(train_queue)))] for k in range(len_batch)]
    # print(train_queue)

    for i, batch_queue in enumerate(train_queue):

        max_place = max([len(train_data[u][id]['loc']) for u,id in batch_queue])

        place = torch.cat([pad_tensor(train_data[u][id]['loc'],max_place,0).unsqueeze(0) for u, id in batch_queue], dim=0).to(device).long()
        time = torch.cat([pad_tensor(train_data[u][id]['tim'],max_place,0).unsqueeze(0) for u, id in batch_queue], dim=0).to(device).long()
        target_place = torch.cat([pad_tensor(train_data[u][id]['target_loc'],max_place,0).unsqueeze(0) for u, id in batch_queue], dim=0)
        target_time = torch.cat([pad_tensor(train_data[u][id]['target_tim'],max_place,0).unsqueeze(0) for u, id in batch_queue], dim=0)

        user = torch.from_numpy(np.array([u for u, _ in batch_queue])).to(device).long()
        print("user", user)

        print("tar place\n",target_place)
        # target = target.to(device).long()
        # target = target.contiguous().view(-1)

        print("place\n",place)
        print("tar time\n",target_time)
        print("time\n",time)
        print("Exit in train)epoch")
        exit()
        


# 读取数据
    #foursquare_dataset = {
    #     'data_neural': self.data_neural,
    #     'vid_list': self.vid_list, 'uid_list': self.uid_list,
    #     'data_filter': self.data_filter,
    #     'vid_lookup': self.vid_list_lookup
#     # }
# dataset_4qs = pickle.load(open('./data/tweets-cikm.txtfoursquare.pk', 'rb'))
#     # data[u][i]: {'history_loc', 'hitory_tim', 'history_count',
#     #              'loc', 'tim', 'target'}    
#     # traj_idx: {u: train_id or test_id}
# train_data, train_traj_idx = generate_input_history(data_neural=dataset_4qs["data_neural"], mode="train")
# # test_data, test_traj_idx = generate_input_history(data_neural=dataset_4qs["data_neural"], mode="test")
# # 位置数
# place_dim = len(dataset_4qs['vid_list']) + 1 
# # train_queue = generate_queue(train_traj_idx, 'random')

# train_epoch(train_data, train_traj_idx, optimizer=None, device='cuda', epoch=5, batch_size=4)
 
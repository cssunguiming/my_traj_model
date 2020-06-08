from collections import Counter
import time
import numpy as np
import pickle

class Data_deepmove(object):
    def __init__(self, trace_min=10, global_visit=10, hour_gap=72,
                 min_gap=20, session_min=3, session_max=10,
                 sessions_min=5, train_split=0.8, embedding_len=50):

        self.path = './data/tweets-cikm.txt'
        self.save_path = './data/'
        self.save_name = 'foursquare'

        self.train_len_min = trace_min
        self.location_global_visit_min = global_visit
        self.hour_gap = hour_gap
        self.min_gap = min_gap
        self.session_max = session_max
        self.filter_short_session = session_min
        self.sessions_count_min = sessions_min
        self.words_embeddings_len = embedding_len

        self.train_split = train_split

        self.data = {}
        self.venues = {}
        self.words_original = []
        self.words_lens = []
        # self.dictionary = dict()
        self.words_dict = None
        self.data_filter = {}
        self.user_filters = None
        self.uid_list = {}
        self.vid_list = {'pad':[0, -1], 'unk':[1, -1]}
        self.vid_list_lookup = {}
        self.pid_loc_lat = {}
        self.data_neural = {}

    # read raw trajectory
    # 0: 9649381944
    # 1: 14911445
    # 2: 40.7555
    # 3: -73.9684
    # 4: 2010-02-25 23:40:19
    # 5: 320283619.0
    # 6: cornerstone tavern 961 second avenue 51st
    # 7: I'm at Cornerstone Tavern (961 Second Avenue at 51st St New York). http://4sq.com/7gLuR2
    # 8: 4a79f20cf964a52013e81fe3
    def load_trajectort_from_dataset(self):
        with open(self.path, encoding='utf-8') as fid:
            for i, line in enumerate(fid):
                _, uid, _, _, tim, _, _, tweet, pid = line.strip('\r\n').split('')
                if uid not in self.data:
                    self.data[uid] = [[pid, tim]]
                else:
                    self.data[uid].append([pid, tim])
                if pid not in self.venues:
                    self.venues[pid] = 1
                else:
                    self.venues[pid] += 1
                # if i>4:
                #     print(self.data)
                #     print(self.venues)
                #     exit()
    def filter_users_by_length(self):
        uid3 = [x for x in self.data if len(self.data[x])>self.train_len_min]
        pick3 = sorted([(x, len(self.data[x])) for x in uid3], key=lambda x: x[1], reverse=True)
        pid3 = [x for x in self.venues if self.venues[x]>self.location_global_visit_min]
        pid_pic3 = sorted([(x, self.venues[x]) for x in pid3], key=lambda x:x[1], reverse=True )
        pid_3 = dict(pid_pic3)

        session_len_list = []
        for u in pick3:
            uid = u[0]
            data = self.data[uid]
            topk = Counter([x[0] for x in data]).most_common()
            topk1 = [x[0] for x in topk if x[1]>1]
            sessions = {}
            for i, record in enumerate(data):
                pid, tim = record
                try:
                    tid = int(time.mktime(time.strptime(tim, "%Y-%m-%d %H:%M:%S")))
                except Exception as e:
                    print('errror:{}'.format(e))
                    continue
                sess_id = len(sessions)
                if pid not in pid_3 and pid not in topk1:
                    continue
                if i==0 or len(sessions)==0:
                    sessions[sess_id] = [record]
                else:
                    if (tid-last_tid)/3600 > self.hour_gap or len(sessions[sess_id-1])>self.session_max:
                        sessions[sess_id] = [record]
                    elif (tid-last_tid)/60 > self.min_gap:
                        sessions[sess_id-1].append(record)
                    else:
                        pass
                last_tid = tid
            sessions_filter = {}
            for s in sessions: 
                if len(sessions[s])>=self.filter_short_session:
                    sessions_filter[len(sessions_filter)] = sessions[s]
                    session_len_list.append(len(sessions[s]))
            if len(sessions_filter)>=self.sessions_count_min:
                self.data_filter[uid] = {
                    'sessions_count': len(sessions_filter),
                    'topk_count': len(topk1),
                    'topk': topk1,
                    'sessions': sessions_filter,
                    'raw_sessions': sessions
                }
        self.user_filter3 = [x for x in self.data_filter if
                             self.data_filter[x]['sessions_count']>=self.sessions_count_min]
    
    def build_users_locations_dict(self):
        for u in self.user_filter3:
            sessions = self.data_filter[u]['sessions']
            if u not in self.uid_list:
                self.uid_list[u] = [len(self.uid_list), len(sessions)]
            for sess_id in sessions:
                pids = [x[0] for x in sessions[sess_id]]
                for p in pids:
                    if p not in self.vid_list:
                        self.vid_list_lookup[len(self.vid_list)] = p
                        self.vid_list[p] = [len(self.vid_list), 1]
                    else:
                        self.vid_list[p][1] += 1
    
    
    @staticmethod
    def tid_list(tmd):
        tm = time.strptime(tmd, "%Y-%m-%d %H:%M:%S")
        tid = tm.tm_wday * 24 + tm.tm_hour
        return tid
    @staticmethod
    def tid_list_48(tmd):
        tm = time.strptime(tmd, "%Y-%m-%d %H:%M:%S")
        if tm.tm_wday in [0, 1, 2, 3, 4]:
            tid = tm.tm_hour
        else:
            tid = tm.tm_hour + 24
        return tid + 1

    def prepare_neural_data(self):

        n_list = []
        
        for u in self.uid_list:

            n = 0

            sessions = self.data_filter[u]['sessions']
            sessions_tran = {} # transform time
            sessions_id = []
            for sess_id in sessions:
                sessions_tran[sess_id] = [[self.vid_list[x[0]][0], self.tid_list_48(x[1]) ] for x in sessions[sess_id]]
                n += len(sessions_tran[sess_id])
                sessions_id.append(sess_id)

            n_list.append(n)
                
            
            split_id = int(np.floor(self.train_split*len(sessions_id)))
            train_id = sessions_id[:split_id]
            test_id = sessions_id[split_id:]
            pred_len = sum([len(sessions_tran[i])-1 for i in train_id])
            valid_len = sum([len(sessions_tran[i])-1 for i in test_id])
            train_loc = {}
            for i in train_id:
                for sess in sessions_tran[i]:
                    if sess[0] in train_loc:
                        train_loc[sess[0]] += 1
                    else:
                        train_loc[sess[0]] = 1
            
            self.data_neural[self.uid_list[u][0]] = {
                'sessions': sessions_tran, 
                'train': train_id, 'test': test_id,
                'pred_len': pred_len, 'valid_len':valid_len,
                'train_loc': train_loc
                # 'explore': location_ratio, 'entropy': entropy, 'rg': rg
            }

        print("max_n",np.mean(n_list))


    def save_variables(self):
        foursquare_dataset = {
            'data_neural': self.data_neural,
            'vid_list': self.vid_list, 'uid_list': self.uid_list,
            # 'parameters': self.get_parameters(),
            'data_filter': self.data_filter,
            'vid_lookup': self.vid_list_lookup
        }
        pickle.dump(foursquare_dataset, open(self.path + self.save_name + '.pk', 'wb'))





data_generator = Data_deepmove()
# parameters = data_generator.get_parameters()
# print('############PARAMETER SETTINGS:\n' + '\n'.join([p + ':' + str(parameters[p]) for p in parameters]))
print('############START PROCESSING:')
print('load trajectory from {}'.format(data_generator.path))
data_generator.load_trajectort_from_dataset()
print('filter users')
data_generator.filter_users_by_length()
print('build users/locations dictionary')
data_generator.build_users_locations_dict()
# data_generator.load_venues()
# data_generator.venues_lookup()
print('prepare data for neural network')
data_generator.prepare_neural_data()
print('save prepared data')
data_generator.save_variables()
print('raw users:{} raw locations:{}'.format(
    len(data_generator.data), len(data_generator.venues)))
print('final users:{} final locations:{}'.format(
    len(data_generator.data_neural), len(data_generator.vid_list)))

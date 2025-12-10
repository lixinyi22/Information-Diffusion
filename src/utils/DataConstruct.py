''' Data Loader class for training iteration '''
import random
import numpy as np
import torch
from torch.autograd import Variable
import utils.Constants as Constants
import logging
import pickle

class Options(object):
    
    def __init__(self, data_name = 'Twitter'):
        #train file path.
        self.train_data = data_name+'/cascade.txt'
        self.train_data_id = data_name+'/cascade_id.txt'
        #valid file path.
        self.valid_data = data_name+'/cascadevalid.txt'
        self.valid_data_id = data_name+'/cascadevalid_id.txt'
        #test file path.
        self.test_data = data_name+'/cascadetest.txt'
        self.test_data_id = data_name+'/cascadetest_id.txt'

        self.u2idx_dict = data_name+'/u2idx.pickle'
        self.ui2idx_dict = data_name+'/ui2idx.pickle'
        self.idx2u_dict = data_name+'/idx2u.pickle'
        self.net_data = data_name+'/edges.txt'
        self.content_embedding = data_name+'/id2embedding.pickle'

def Split_data(data_name, with_EOS=True, max_len = 500):
    options = Options(data_name)
    _u2idx = {}
    #idx2u = []
    
    with open(options.u2idx_dict, 'rb') as handle:
        _u2idx = pickle.load(handle)
    with open(options.ui2idx_dict, 'rb') as handle:
        _ui2idx = pickle.load(handle)
    with open(options.content_embedding, 'rb') as handle:
        _content_embedding = pickle.load(handle)
    user_size = len(_u2idx)
    
    def build_dataset(dataset_path: str):
        t_cascades = []
        timestamps = []
        max_time, min_time = 0, 1000000000000
    
        for line in open(dataset_path):
            if len(line.strip()) == 0:
                continue
            timestamplist = []
            userlist = []
            chunks = line.strip().split()
            for chunk in chunks:
                try:
                    user, timestamp = chunk.split(",")
                except:
                    pass
                if user in _u2idx:
                    userlist.append(_u2idx[user])
                    timestamplist.append(float(timestamp))
                    if float(timestamp) > max_time:
                        max_time = float(timestamp)
                    if float(timestamp) < min_time:
                        min_time = float(timestamp)
            
            if len(userlist) > max_len:    
                userlist = userlist[:max_len]
                timestamplist = timestamplist[:max_len]
                
            if len(userlist) >= 1:
                if with_EOS:
                    userlist.append(Constants.EOS)
                    timestamplist.append(Constants.EOS)
                t_cascades.append(userlist)
                timestamps.append(timestamplist)
                
        '''ordered by timestamps'''        
        order = [i[0] for i in sorted(enumerate(timestamps), key=lambda x:x[1])]
        timestamps = sorted(timestamps)
        t_cascades[:] = [t_cascades[i] for i in order]
        return t_cascades, timestamps, max_time, min_time
    
    def build_content_embedding(dataset_path: str):
        content_embedding = []
        info_id_list = []
        for line in open(dataset_path):
            if len(line.strip()) == 0:
                continue
            info_id = int(line.strip())
            content_embedding.append(_content_embedding[info_id])
            info_id_list.append(info_id)
        return content_embedding, info_id_list
    
    '''data split'''
    max_time, min_time = 0, 1000000000000
    train, train_t, train_max_time, train_min_time = build_dataset(options.train_data)
    if train_max_time > max_time:
        max_time = train_max_time
    if train_min_time < min_time:
        min_time = train_min_time
    valid, valid_t, valid_max_time, valid_min_time = build_dataset(options.valid_data)
    if valid_max_time > max_time:
        max_time = valid_max_time
    if valid_min_time < min_time:
        min_time = valid_min_time
    test, test_t, test_max_time, test_min_time = build_dataset(options.test_data)
    if test_max_time > max_time:
        max_time = test_max_time
    if test_min_time < min_time:
        min_time = test_min_time
    train_content_embedding, train_id = build_content_embedding(options.train_data_id)
    valid_content_embedding, valid_id = build_content_embedding(options.valid_data_id)
    test_content_embedding, test_id = build_content_embedding(options.test_data_id)
    info_size = len(train_content_embedding)
    
    train = [train, train_t, train_content_embedding, train_id]
    valid = [valid, valid_t, valid_content_embedding, valid_id]
    test = [test, test_t, test_content_embedding, test_id]
    
    t_cascades = train[0] + valid[0] + test[0]
    total_len =  sum(len(i)-1 for i in t_cascades)
    train_size = len(train_t)
    valid_size = len(valid_t)
    test_size = len(test_t)
    logging.info(f"training size: {train_size}\n   valid size: {valid_size}\n  testing size: {test_size}")
    logging.info(f"total size: {len(t_cascades)}")
    logging.info(f"average length: {total_len/len(t_cascades)}")
    logging.info(f'maximum length: {max(len(cas) for cas in t_cascades)}')
    logging.info(f'minimum length: {min(len(cas) for cas in t_cascades)}')    
    logging.info(f"user size:%d"%(user_size-2))           
    
    return user_size, info_size, train, valid, test, max_time-min_time

def LoadContentEmbedding(data_name):
    options = Options(data_name)
    with open(options.content_embedding, 'rb') as handle:
        content_embedding_dict = pickle.load(handle)
        
    content_embedding = []
    for line in open(options.train_data_id):
        if len(line.strip()) == 0:
            continue
        info_id = int(line.strip())
        content_embedding.append(content_embedding_dict[info_id])
        
    content_embedding = torch.tensor(content_embedding, dtype=torch.float32)
    return content_embedding

class DataConstruct(object):
    ''' For data iteration '''

    def __init__(self, data, batch_size, cuda=True, shuffle=False, test=False, with_EOS=True, max_len=500):
        self._batch_size = batch_size
        self.cas = data[0]
        self.time = data[1]
        self.content_embedding = data[2]
        self.original_idx = data[3]
        self.idx = [x for x in range(0, len(self.cas))]
        self.test = test
        self.with_EOS = with_EOS
        self.cuda = cuda
        self.max_len = max_len

        self._n_batch = int(np.ceil(len(self.cas) / self._batch_size))
        self._iter_count = 0
        self._need_shuffle = shuffle

        if self._need_shuffle:
            num = [x for x in range(0, len(self.cas))]
            random_seed_int = random.randint(0, 1000)
            logging.info(f"Init Dataset and shuffle data with {random_seed_int}")
            random.seed(random_seed_int)
            random.shuffle(num)
            self.cas = [self.cas[num[i]] for i in range(0, len(num))]
            self.time = [self.time[num[i]] for i in range(0, len(num))]
            self.content_embedding = [self.content_embedding[num[i]] for i in range(0, len(num))]
            self.idx = [self.idx[num[i]] for i in range(0, len(num))]
            self.original_idx = [self.original_idx[num[i]] for i in range(0, len(num))]
            
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        def pad_to_longest(insts):
            ''' Pad the instance to the max seq length in batch '''

            #max_len = max(len(inst) for inst in insts)
            max_len = self.max_len + 1
            inst_data = np.array([
                inst + [Constants.PAD] * (max_len - len(inst))
                for inst in insts])
        
            inst_data_tensor = Variable(
                torch.LongTensor(inst_data), volatile=self.test)

            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()

            return inst_data_tensor

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            seq_insts = self.cas[start_idx:end_idx]
            seq_timestamp = self.time[start_idx:end_idx]
            seq_content_embedding = torch.tensor(self.content_embedding[start_idx:end_idx], dtype=torch.float32)
            seq_id = torch.LongTensor(self.idx[start_idx:end_idx])
            #seq_id = torch.LongTensor(self.original_idx[start_idx:end_idx])
            seq_data = pad_to_longest(seq_insts)
            seq_data_timestamp = pad_to_longest(seq_timestamp)
            
            return seq_data, seq_data_timestamp, seq_content_embedding, seq_id
        else:

            if self._need_shuffle:
                num = [x for x in range(0, len(self.cas))]
                random_seed_int = random.randint(0, 1000)
                logging.info(f"shuffle data with {random_seed_int}")
                random.seed(random_seed_int)
                random.shuffle(num)
                self.cas = [self.cas[num[i]] for i in range(0, len(num))]
                self.time = [self.time[num[i]] for i in range(0, len(num))]
                self.content_embedding = [self.content_embedding[num[i]] for i in range(0, len(num))]
                self.idx = [self.idx[num[i]] for i in range(0, len(num))]
                self.original_idx = [self.original_idx[num[i]] for i in range(0, len(num))]

            self._iter_count = 0
            raise StopIteration()

import os
import pickle as pkl
import numpy as np

from collections import namedtuple
from sklearn.preprocessing import LabelBinarizer

def load_pkl(filename):
    f=open(filename,'rb')
    obj=pkl.load(f)
    f.close()
    return obj

def dump_pkl(obj,filename):
    f=open(filename,'wb')
    pkl.dump(obj,f)
    f.close()
    return obj

def load_vocabs(filename):
    with open(filename,encoding='utf8') as f:
        id2word=[line.rstrip().split(' ')[0] for line in f]
    word2id={}
    for i,word in enumerate(id2word): 
        word2id[word]=i
    return id2word,word2id

def load_train_ids_pairs(filename,eos_id=2,reverse=False):
    '''自动在训练集response后加上<\s>标记
    '''
    ids_pairs=[]
    with open(filename,encoding='utf8') as f:
        for line in f:
            pair=line.rstrip().split('\t')
            post=pair[0]
            response=pair[1].rstrip()
            near=pair[2]##############
            if reverse:
                post_ids=[int(c) for c in response.split(' ')]
                response_ids=[int(c) for c in post.split(' ')]+[eos_id]
            else:
                post_ids=[int(c) for c in post.split(' ')]
                response_ids=[int(c) for c in response.split(' ')]+[eos_id]
                near_ids=[int(c) for c in near.split(' ')]##################
            ids_pairs.append((post_ids,response_ids,near_ids))###################
    return ids_pairs



def load_eval_ids_pairs(filename):
    results=[]
    with open(filename,encoding='utf8') as f:
        for line in f:
            pair=line.rstrip().split('\t')
            pair[0]=pair[0]
            pair[1]=pair[1]
            pair[2]=pair[2].rstrip()
            post_words=[c for c in pair[0].split(' ')]
            post_ids=[int(c) for c in pair[1].split(' ')]
            response_words_list=[[c for c in pair[2].split(' ')]]###################
            near_ids=[int(c) for c in pair[3].split(' ')]###################
            #response_words_list=[[c for c in p.split(' ')] for p in pair[3:]]##############
            results.append([post_words,post_ids,response_words_list,near_ids])###############
    return results

class STCWeiboDataset(object):
    def __init__(self, dir_path,use_buffer=True,buffer_name='buffer.pkl',reverse=False,location_clf=False):
        '''
        reverse: 是否使用response生成post
        '''
        self.dir_path=dir_path
        buffer_filename=os.path.join(dir_path,buffer_name)
        self.reverse=reverse
        self.location_clf=location_clf
        if use_buffer and os.path.exists(buffer_filename):
            tmp_dict=load_pkl(buffer_filename)
            self.__dict__.update(tmp_dict)
            print('已从缓存载入')
        else:
            self.load_data()
            buffer_filename=os.path.join(dir_path,buffer_name)
            dump_pkl(self.__dict__,buffer_filename)
            print('已创建缓存')
    
    def load_data(self):
        if self.reverse:
            post_weight_filename='response_weights.pkl'
            response_weight_filename='post_weights.pkl'
            post_vocabs_filename='response_vocabs.txt'
            response_vocabs_filename='post_vocabs.txt'
        else:
            post_weight_filename='post_weights.pkl'
            response_weight_filename='response_weights.pkl'
            post_vocabs_filename='post_vocabs.txt'
            response_vocabs_filename='response_vocabs.txt'
            
        self.post_weights=load_pkl(os.path.join(self.dir_path,post_weight_filename))
        self.response_weights=load_pkl(os.path.join(self.dir_path,response_weight_filename))
        self.post_vocabs,self.post_dict=load_vocabs(os.path.join(self.dir_path,post_vocabs_filename))
        self.response_vocabs,self.response_dict=load_vocabs(os.path.join(self.dir_path,response_vocabs_filename))
        self.eos_id=self.post_dict['<\s>']
        self.pad_id=self.post_dict['<pad>']
        self.unk_id=self.post_dict['<unk>']
            
        if self.location_clf==False:    
            self.train_ids=load_train_ids_pairs(os.path.join(self.dir_path,'train_ids.txt'),reverse=self.reverse)
            self.valid_ids=load_eval_ids_pairs(os.path.join(self.dir_path,'valid_ids.txt'))
            self.eval_ids=load_eval_ids_pairs(os.path.join(self.dir_path,'eval_ids.txt'))
        else:
            self.train_ids=self.load_cls_train_pairs(os.path.join(self.dir_path,'cls_result.train'))
            self.valid_ids=load_eval_ids_pairs(os.path.join(self.dir_path,'valid_ids.txt'))
            self.eval_ids=load_eval_ids_pairs(os.path.join(self.dir_path,'eval_ids.txt'))

        
        self.train_cnt=len(self.train_ids)
        self.valid_cnt=len(self.valid_ids)
        self.eval_cnt=len(self.eval_ids)
        self.shuffle_idx=np.arange(self.train_cnt)

        assert(self.eos_id==self.response_dict['<\s>'])
        
    def refresh_shuffle_idx(self):
        self.cur_num=0
        np.random.shuffle(self.shuffle_idx)
        
    def get_train_batch(self,batch_size=32):
        self.refresh_shuffle_idx()
        while self.cur_num<self.train_cnt:
            batch_idx=self.shuffle_idx[self.cur_num:self.cur_num+batch_size]
            batch_data=[self.train_ids[i] for i in batch_idx]
            self.cur_num+=batch_size
            self.actual_batch_size=len(batch_data)
            batch_data=self._list_transpose(batch_data)
            post_input=self._padding(batch_data[0])
            response_input=self._padding(batch_data[1])
            near_input=self._padding(batch_data[2])#####################
            if self.location_clf:
                label_input=np.array(batch_data[2])
                yield post_input,response_input,label_input
            else:
                yield post_input,response_input,near_input#####################
            
    def get_one_train_batch(self,batch_size=32):
        for data in self.get_train_batch(batch_size):
            break
        return data
    
    def _list_transpose(self,data):
        result=[]
        if len(data)==0:
            return result
        total=len(data[0])
        
        for i in range(total):
            result.append([item[i] for item in data])
        return result
    
    def _padding(self,data):
        max_len=np.max([len(item) for item in data])
        result=[]
        for item in data:
            if len(item)<max_len:
                result.append(item+[self.pad_id for i in range(max_len-len(item))])
            else:
                result.append(item)
        return np.array(result)
        
    def _get_eval_batch(self,data,batch_size,total):
        cnt=0
        while cnt<total:
            batch_data=data[cnt:cnt+batch_size]
            cnt+=batch_size
            post_words=[item[0] for item in batch_data]
            post_ids=[item[1] for item in batch_data]
            response_list=[item[2] for item in batch_data]
            near_ids=[item[3] for item in batch_data]#############################
            yield post_words,self._padding(post_ids),response_list,self._padding(near_ids)##################
            #yield self._padding(post_ids),near_ids##################
    
    def get_eval_batch(self,batch_size=32):
        return self._get_eval_batch(self.eval_ids,batch_size,self.eval_cnt)
    
    def get_valid_batch(self,batch_size=32):
        return self._get_eval_batch(self.valid_ids,batch_size,self.valid_cnt)
    
    def load_cls_train_pairs(self,filename,):
        '''自动在训练集response后加上<\s>标记
        '''
        self.location_labels='华北,华东,华南,西南,华中,东北,西北,海外,无信息'.split(',')
        self.location_dict=dict([(item,i) for i,item in enumerate(self.location_labels)])
        ids_pairs=[]
        with open(filename,encoding='utf8') as f:
            for line in f:
                pair=line.rstrip().split('\t')

                post_item=pair[:3]
                resp_item=pair[3:]
                post_dict=self.post_dict
                response_dict=self.response_dict

                resp_label=resp_item[0]
                resp_prob=[float(v) for v in resp_item[1].split(',')]
                response=resp_item[2]
                
                post_ids=[post_dict.get(c,self.unk_id) for c in post_item[2].split(' ')]
                response_ids=[response_dict.get(c,self.unk_id) for c in response.split(' ')]+[self.eos_id]
                ids_pairs.append((post_ids,response_ids,resp_prob,))
        return ids_pairs
    
   

if __name__=='__main__':
    db=STCWeiboDataset(dir_path='Untitled Folder/')
    
import os,sys
sys.path.append(os.path.abspath('../src'))

from datasets_new import STCWeiboDataset

%%time
#载入数据
db=STCWeiboDataset('final2/',use_buffer=False)

unk_id=db.post_dict['<unk>']
bos_id=db.post_dict['<s>']
eos_id=db.post_dict['<\s>']
pad_id=db.post_dict['<pad>']

#训练数据样例
batch_data=db.get_one_train_batch(128)

batch_data[2][4]

import os
import tensorflow as tf 
gpu_id='3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']=gpu_id
config=tf.ConfigProto()
config.gpu_options.allow_growth=True

from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.distributions import bernoulli
from tensorflow.python.ops.distributions import categorical
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq.python.ops.helper import _unstack_ta,_transpose_batch_time,Helper


import collections
from tensorflow.python.ops.rnn_cell_impl import RNNCell,_Linear

_EncryptLSTMStateTuple = collections.namedtuple("EncryptLSTMStateTuple", ("c", "h","e"))

class EncryptLSTMStateTuple(_EncryptLSTMStateTuple):
    __slots__ = ()

    @property
    def dtype(self):
        (c, h, e) = self
        return c.dtype


class EncryptLSTMCell(RNNCell):
    def __init__(self, num_units, forget_bias=1.0,
               state_is_tuple=True, activation=None, reuse=None):

        super(EncryptLSTMCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh
        self._linear = None
        self._linear1 = None

    @property
    def state_size(self):
        return EncryptLSTMStateTuple(self._num_units, self._num_units, self._num_units)
                

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        sigmoid = math_ops.sigmoid
        # Parameters of gates are concatenated into one multiply for efficiency.
        c, h, e = state
        
        if self._linear is None:
            self._linear = _Linear([inputs, h], 5 * self._num_units, True)
            #self._linear = _Linear([inputs, h], 4 * self._num_units, True)
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o ,arf= array_ops.split(
            value=self._linear([inputs, h]), num_or_size_splits=5, axis=1)

        with tf.variable_scope('mapeh'):
            if self._linear1 is None:
                self._linear1 = _Linear([e, h],  self._num_units, True)
            e1=array_ops.split(
                value=self._linear1([e,h]), num_or_size_splits=1, axis=0)[0]
            
        new_c = (
            c * sigmoid(f + self._forget_bias) + sigmoid(i) * (sigmoid(arf)*(self._activation(j))+(1-sigmoid(arf))*(self._activation(e1)))
        )
        new_h = self._activation(new_c) * sigmoid(o)         
            
        new_state = EncryptLSTMStateTuple(new_c, new_h, e)
        
        return new_h, new_state
    
    def initial_state(self,batch_size,dtype,encrypt_state):
        state=self.zero_state(batch_size,dtype)
        return EncryptLSTMStateTuple(state.c,state.h,encrypt_state)

## 用于训练的模型

from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, DropoutWrapper
from seq2seq_helper import OutputWrapper,fix_beam_search_bug
tf.reset_default_graph()
max_length=300
hidden_size=1000
vocab_size=20004
embedding_size=620

batch_size=tf.placeholder(tf.int32,shape=[],name='batch_size')
keep_prob_placeholder=tf.placeholder(tf.float32,name='keep_prob_placeholder')
encoder_inputs=tf.placeholder(tf.int32,[None,None],name='encoder_inputs')
decoder_targets=tf.placeholder(tf.int32,[None,None],name='decoder_targets')
rencoder_inputs=tf.placeholder(tf.int32,[None,None],name='rencoder_inputs')
max_gradient_norm=5.0

def compute_mask(inputs):
    #compute mask
    mask=tf.not_equal(inputs,pad_id)
    inputs_length=tf.reduce_sum(tf.cast(mask,dtype=tf.int32),axis=-1,name='inputs_length')
    mask=tf.cast(mask,dtype=tf.float32)
    return inputs_length,mask


with tf.variable_scope('encoder',reuse=False):
    encoder_inputs_length,_=compute_mask(rencoder_inputs)
    encoder_embedding=tf.Variable(db.post_weights,name='embedding',dtype=tf.float32)

    encoder_inputs_embedded=tf.nn.embedding_lookup(encoder_embedding,rencoder_inputs)

    encoder_cell=LSTMCell(hidden_size)
    encoder_cell=DropoutWrapper(encoder_cell,output_keep_prob=keep_prob_placeholder)

    encoder_outputs,encoder_state=tf.nn.dynamic_rnn(encoder_cell,
                                                  inputs=encoder_inputs_embedded,
                                                  sequence_length=encoder_inputs_length,
                                                  dtype=tf.float32)
    rq_vector=encoder_outputs[:,-1,:]



with tf.variable_scope('encrypter'):
    encoder_inputs_length,_=compute_mask(encoder_inputs)
    encoder_embedding=tf.Variable(db.post_weights,name='embedding',dtype=tf.float32)

    encoder_inputs_embedded=tf.nn.embedding_lookup(encoder_embedding,encoder_inputs)

    #encoder_cell=EncryptLSTMCell(hidden_size)
    lstm_fw_cell = EncryptLSTMCell(hidden_size)
    lstm_bw_cell = EncryptLSTMCell(hidden_size)    
    #initial_state=encoder_cell.initial_state(batch_size,dtype=tf.float32,encrypt_state=rq_vector)
    
    initial_state_fw = lstm_fw_cell.zero_state(batch_size, dtype=tf.float32)
    initial_state_bw = lstm_bw_cell.zero_state(batch_size, dtype=tf.float32)
    
    
    #encoder_cell=DropoutWrapper(encoder_cell,output_keep_prob=keep_prob_placeholder)
    lstm_fw_cell=DropoutWrapper(lstm_fw_cell,output_keep_prob=keep_prob_placeholder)
    lstm_bw_cell=DropoutWrapper(lstm_bw_cell,output_keep_prob=keep_prob_placeholder)
    
    
    ((encoder_fw_outputs,
              encoder_bw_outputs),
             (encoder_fw_state,
              encoder_bw_state))=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                                   lstm_bw_cell,
                                                                   initial_state_fw=initial_state_fw,
                                                                   initial_state_bw=initial_state_bw,
                                                   inputs=encoder_inputs_embedded,
                                                  sequence_length=encoder_inputs_length,
                                                  dtype=tf.float32)
    
    encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs),2)
    #encoder_state = tf.concat((encoder_fw_state, encoder_bw_state),-1)
    #encoder_outputs = (encoder_fw_outputs, encoder_bw_outputs)
    encoder_state = (encoder_fw_state, encoder_bw_state)
    e1 = tf.add(encoder_state[0].c, encoder_state[1].c)
    e2 = tf.add(encoder_state[0].h, encoder_state[1].h)
    e3 = tf.add(encoder_state[0].e, encoder_state[1].e)
    encoder_state=EncryptLSTMStateTuple(e1,e2,e3)
    #encoder_state=encoder_fw_state
with tf.variable_scope('decoder'):
    #decoder_initial_state_fw=EncryptLSTMStateTuple(encoder_fw_state.c,encoder_fw_state.h,rq_vector)
    #decoder_initial_state_bw=EncryptLSTMStateTuple(encoder_bw_state.c,encoder_bw_state.h,rq_vector)
    #e1 = tf.concat((encoder_state[0].c, encoder_state[1].c),1)
    #e2 = tf.concat((encoder_state[0].h, encoder_state[1].h),1)
    #decoder_initial_state=EncryptLSTMStateTuple(e1,e2,rq_vector)
    decoder_initial_state=EncryptLSTMStateTuple(encoder_state.c,encoder_state.h,rq_vector)
    #decoder_initial_state=encoder_state
    bos=tf.ones_like(decoder_targets[:,:1],dtype=tf.int32)*bos_id
    decoder_inputs=tf.concat([bos,decoder_targets[:,:-1]],axis=-1,name='decoder_inputs')
    decoder_targets_length,decoder_target_mask=compute_mask(decoder_targets)

    decoder_embedding=tf.Variable(db.response_weights,name='embedding',dtype=tf.float32)

    decoder_targets_embedded=tf.nn.embedding_lookup(decoder_embedding,decoder_inputs)

    decoder_cell=EncryptLSTMCell(hidden_size)
    #decoder_cell_fw=EncryptLSTMCell(hidden_size)
    #decoder_cell_bw=EncryptLSTMCell(hidden_size)
    #decoder_cell=LSTMCell(hidden_size)
    decoder_cell=DropoutWrapper(decoder_cell,output_keep_prob=keep_prob_placeholder)

    training_helper=tf.contrib.seq2seq.TrainingHelper(decoder_targets_embedded,decoder_targets_length,time_major=False,)

    output_layer1 = tf.layers.Dense(vocab_size,kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
    output_layer = OutputWrapper(output_layer1,name='output_wrapper',infer=False)
    decoder=tf.contrib.seq2seq.BasicDecoder(decoder_cell,training_helper,decoder_initial_state,output_layer=output_layer)
    decoder_outputs,_,_=tf.contrib.seq2seq.dynamic_decode(decoder,maximum_iterations=max_length,impute_finished=True)

    

#encoder_outputs,encoder_state=do_encode(decoder_targets,q_vector)

#decoder_outputs,decoder_target_mask=do_decode(encoder_state,q_vector,decoder_targets)

decoder_logits_train=tf.identity(decoder_outputs.rnn_output)
decoder_predict_train=tf.argmax(decoder_logits_train,axis=-1,name='decoder_predict_train')

cos_weight=0.6

loss=tf.contrib.seq2seq.sequence_loss(logits=decoder_logits_train,targets=decoder_targets,weights=decoder_target_mask)

tf.summary.scalar('loss',loss)
optimizer=tf.train.AdamOptimizer()
trainable_params=tf.trainable_variables()
gradients=tf.gradients(loss,trainable_params)
clip_gradients,_=tf.clip_by_global_norm(gradients,max_gradient_norm)
train_op=optimizer.apply_gradients(zip(clip_gradients,trainable_params))
    
tf.summary.scalar('loss', loss)
summary_op = tf.summary.merge_all()


def train(sess,batch):
    feed_dict={encoder_inputs:batch[0],
               decoder_targets:batch[1],
               rencoder_inputs:batch[2],
              keep_prob_placeholder:0.5,
               batch_size:len(batch[0])
              }
    _,_loss,_summary_op=sess.run([train_op,loss,summary_op],feed_dict=feed_dict)
    return _loss,_summary_op

## 预测时的代码 BeamSearch

from tensorflow.python.layers import base
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import nn
from tensorflow.python.eager import context

class SkipDense(base.Layer):
    def __init__(self,dense,unknown_id=0,name=None,**kwargs):
        super(SkipDense,self).__init__(trainable=False,name=name,**kwargs)
        self.dense=dense
        self.units=self.dense.units
        self.unknown_id=unknown_id
        
    def build(self,input_shape):
        kernel_mask=np.ones(shape=(1000,20004),dtype=np.float32)
        kernel_mask[:,self.unknown_id]=0
        bias_mask=np.ones(shape=(20004,),dtype=np.float32)
        bias_mask[self.unknown_id]=0
        kernel_mask_tensor=tf.constant(kernel_mask,dtype=tf.float32)
        bias_mask_tensor=tf.constant(bias_mask,dtype=tf.float32)
        
        self.kernel=self.dense.kernel*kernel_mask_tensor
        self.bias=self.dense.bias*bias_mask_tensor
        
        self.built=True
        
    def call(self,inputs):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        shape = inputs.get_shape().as_list()
        if len(shape) > 2:
          # Broadcasting is required for the inputs.
          outputs = standard_ops.tensordot(inputs, self.kernel, [[len(shape) - 1],
                                                                 [0]])
          # Reshape the output back to the original ndim of the input.
          if context.in_graph_mode():
            output_shape = shape[:-1] + [self.units]
            outputs.set_shape(output_shape)
        else:
            outputs = standard_ops.matmul(inputs, self.kernel)
        
        outputs = nn.bias_add(outputs, self.bias)
        return outputs
    
    def _compute_output_shape(self,input_shape):
        return self.dense._compute_output_shape(input_shape)

import numpy as np
from seq2seq_helper import OutputWrapper,fix_beam_search_bug
beam_size=5

start_tokens = tf.ones((batch_size,)*bos_id,dtype=tf.int32)
end_token = 2

with tf.variable_scope('beam_search'):
    # 如果使用beam_search，则需要将encoder的输出进行tile_batch，其实就是复制beam_size份。
    encoder_beam_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=beam_size)
    encoder_beam_state = tf.contrib.framework.nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, beam_size), encoder_state)
    
    skip_output_layer = SkipDense(output_layer)
    
    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell, embedding=decoder_embedding,
                                                             start_tokens=start_tokens, end_token=end_token,
                                                             initial_state=encoder_beam_state,
                                                             beam_width=beam_size,
                                                             output_layer=skip_output_layer,
                                                             )

    inference_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                    maximum_iterations=max_length)
    predicted_ids=fix_beam_search_bug(inference_decoder_outputs)

    
def infer(sess,batch):
    feed_dict={encoder_inputs:batch[1],
           rencoder_inputs:batch[3],
          keep_prob_placeholder:1.0,
           batch_size:len(batch[1])
          }
    
    predict=sess.run([predicted_ids],feed_dict=feed_dict)
    return predict

import numpy as np
#result=infer(sess,eval_batch[1])

def show_results(posts,result):
    result=list(np.transpose(result[0],[0,2,1]))
    for post,ps in zip(posts,result):
        print('-----')
        print('微博:',''.join(post))
        for i,p in enumerate(ps):
            print('\t回复%d：'%(i+1),''.join([db.response_vocabs[i] for i in p if i >0]))    
            
import jieba
def sentences2ids(sentences):
    ids=[]
    for sentence in sentences:
        words = sentence.split(' ')
        vals=[db.post_dict.get(word,unk_id) for word in words]
        ids.append(vals)
    ids=db._padding(ids)
    return ids

def answer(question):
    q=' '.join(jieba.lcut(question))
    batch=sentences2ids([q])

    result=infer(sess,batch)
    show_results([question],result)

## 评价指标

from gensim.models import Word2Vec

word2vec_model=Word2Vec.load('wiki.en.text.model')

import nltk
from nltk.translate.bleu_score import SmoothingFunction,sentence_bleu
from scipy.spatial.distance import cosine

chencherry=SmoothingFunction()

def bleu(references,hyp):
    try:
        bleu1=sentence_bleu(references, hyp,weights=(1,),smoothing_function=chencherry.method1)
    except:
        bleu1=np.NAN
    try:
        bleu2=sentence_bleu(references, hyp,weights=(0.5,0.5),smoothing_function=chencherry.method1)
    except:
        bleu2=np.NAN
    try:
        bleu3=sentence_bleu(references, hyp,weights=(0.3333,0.3333,0.3333),smoothing_function=chencherry.method1)
    except:
        bleu3=np.NAN
    try:
        sbleu1=sentence_bleu(references, hyp,weights=(1,),smoothing_function=chencherry.method7)
    except:
        sbleu1=np.NAN
    try:
        sbleu2=sentence_bleu(references, hyp,weights=(0.5,0.5),smoothing_function=chencherry.method7)
    except:
        sbleu2=np.NAN
    try:
        sbleu3=sentence_bleu(references, hyp,weights=(0.3333,0.3333,0.3333),smoothing_function=chencherry.method7)
    except:
        sbleu3=np.NAN
    return [bleu1,bleu2,bleu3,sbleu1,sbleu2,sbleu3]

def get_wordemb_score(references,hyp,wv):
    scores=[]
    for ref in references:
        rvectors=[wv[w] for w in ref if w in wv]
        hvectors=[wv[w] for w in hyp if w in wv]
        if len(rvectors)==0 or len(hvectors)==0:
            score=0
        else:
            rvector=np.average(rvectors,axis=0)
            hvector=np.average(hvectors,axis=0)
            score=1-cosine(rvector,hvector)
        scores.append(score)
    return np.max(scores)
def get_distinct(sentences,n=2):
    total=0
    total2=0
    ans=0.0
    ans2=0.0
    word_dict={}
    word_dict2={}
    for sentence in sentences:
        for i in range(len(sentence)-n+1):
            term='_*_'.join(sentence[i:i+n])
            total+=1
            word_dict[term]=word_dict.get(term,0)+1
    n2=1
    for sentence in sentences:
        for i in range(len(sentence)-n2+1):
            term='_*_'.join(sentence[i:i+n2])
            total2+=1
            word_dict2[term]=word_dict2.get(term,0)+1
    if total==0:
        ans=0.0
    else:
        ans=len(word_dict)/total
    if total2==0:
        ans2=0.0
    else:
        ans2=len(word_dict2)/total2
    #print(total)
    #print(total2)
    #return [len(word_dict)/total,len(word_dict)]
    return [ans,ans2]
def get_scores(references,hyp,wv):
    bleu_scores=bleu(references,hyp)
    wordemb_score=get_wordemb_score(references,hyp,wv)
    #dist=get_distinct(hyp,2)
    return np.array([wordemb_score]+bleu_scores)

def evaluate(db,infer):
    names=['wordemb_score','bleu1','bleu2','bleu3','sbleu1','sbleu2','sbleu3']
    eval_score=np.zeros(shape=(7,))
    for eval_batch in db.get_eval_batch(64):
        #print(eval_batch)
        #bbaatt=[]
        #bbaatt.append(eval_batch[1])
        #bbaatt.append(eval_batch[3])
        #print(type(eval_batch[1]))
        #print(type(eval_batch[3]))
        result=infer(sess,eval_batch)
        result=list(np.transpose(result[0],[0,2,1]))

        for references,item in zip(eval_batch[2],result):
            best_hyp=item[0]
            best_hyp=[db.response_vocabs[i] for i in best_hyp if i>3]
            score=get_scores(references,best_hyp,word2vec_model.wv)
            eval_score+=score
    eval_score/=db.eval_cnt
    result_dict=dict(list(zip(names,eval_score)))
    return result_dict

def write_eval_result(writer,result_dict,cnt):
    for name in result_dict:
        value=result_dict[name]
        summary=tf.Summary()
        summary_value=summary.value.add()
        summary_value.simple_value=value
        summary_value.tag=name
        writer.add_summary(summary,cnt)

import random
import numpy as np
f=open('./final1/eval_now.txt','r',encoding='utf-8')
ids_pairs=[]
for line in f:
    pair=line.rstrip().split('\t')
    post_ids=[int(c) for c in pair[0].rstrip().split()]
    response_ids=[int(c) for c in pair[1].rstrip().split()]+[2]
    post_idsx=[int(c) for c in pair[2].rstrip().split()]      
    ids_pairs.append((post_ids,response_ids,post_idsx))
valid_pairs=ids_pairs
def get_valid_pair_batch(valid_pairs,batch_size):
    total=len(valid_pairs)
    cnt=0
    while cnt<total:
        batch_data=valid_pairs[cnt:cnt+batch_size]
        cnt+=len(batch_data)
        batch_data=db._list_transpose(batch_data)
        post_input=db._padding(batch_data[0])
        response_input=db._padding(batch_data[1])
        near_input=db._padding(batch_data[2])
        yield post_input,response_input,near_input


def get_batch_loss_func(sess,batch):
    feed_dict={
            encoder_inputs:batch[0],
               decoder_targets:batch[1],
               rencoder_inputs:batch[2],
              keep_prob_placeholder:0.5,
               batch_size:len(batch[0])
          }
    loss_val=sess.run([loss],feed_dict=feed_dict) 
    return loss_val[0]

def get_valid_loss(valid_pairs,batch_size=64):
    loss_val=0
    for valid_batch in get_valid_pair_batch(valid_pairs,batch_size):
        loss_val+=get_batch_loss_func(sess,valid_batch)*len(valid_batch[0])
    return loss_val/len(valid_pairs)

def write_valid_result(writer,result_dict,cnt):
    for name in result_dict:
        value=result_dict[name]
        summary=tf.Summary()
        summary_value=summary.value.add()
        summary_value.simple_value=value
        summary_value.tag=name
        writer.add_summary(summary,cnt)

## 模型初始化及运行
if __name__ == '__main__':
    saver=tf.train.Saver()
    sess=tf.InteractiveSession(config=config)
    sess.run(tf.global_variables_initializer())

    cnt=0
    model_filename='ckpt/ex505050.ckpt'
    if os.path.exists(model_filename+'-%s'%cnt):
        saver.restore(sess,model_filename+'-%s'%cnt)

    train_writer=tf.summary.FileWriter('xinlogs2/ex505050/train',sess.graph)
    eval_writer=tf.summary.FileWriter('xinlogs2/ex505050/eval',sess.graph)
    valid_writer=tf.summary.FileWriter('xinlogs2/ex505050/valid',sess.graph)

#writer=tf.summary.FileWriter('logs/',sess.graph)

# 使用几条验证数据作为观察对象
    for eval_batch in db.get_eval_batch(32):
        break

    best_valid_loss=100000
    valid_result=dict()
    valid_result['loss']=100
    fl=False
    for k in range(20000):
        for i,batch_data in enumerate(db.get_train_batch(64)):
            _loss,_summary_op=train(sess,batch_data)
            train_writer.add_summary(_summary_op,cnt)
            #writer.add_summary(_summary_op,cnt)
            if i%100==0:
                print(cnt,_loss)
            if cnt%100==0:
                #saver.save(sess,'ckpt/ex13.ckpt',global_step=cnt)
                eval_result=evaluate(db,infer)
                write_eval_result(eval_writer,eval_result,cnt)
                print(eval_result)
                now_valid_loss=get_valid_loss(valid_pairs)
                if now_valid_loss<best_valid_loss:
                    best_valid_loss=now_valid_loss
                    #best_stop=cnt
                    saver.save(sess,'ckpt/ex505050.ckpt',global_step=cnt)
                valid_result['loss']=now_valid_loss
                print(now_valid_loss)
                write_valid_result(valid_writer,valid_result,cnt) 
                print('saved',cnt)
            cnt+=1
            if cnt==100000:
                fl=True
                break
        if fl==True:
            break

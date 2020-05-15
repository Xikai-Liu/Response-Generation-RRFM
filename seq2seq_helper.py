import numpy as np
import tensorflow as tf

def gather_tree_py(values, parents):
    """Gathers path through a tree backwards from the leave nodes. Used
    to reconstruct beams given their parents."""
    beam_length = values.shape[0]
    num_beams = values.shape[1]
    res = np.zeros_like(values)
    res[-1, :] = values[-1, :]
    for beam_id in range(num_beams):
        parent = parents[-1][beam_id]
        for level in reversed(range(beam_length - 1)):
            res[level, beam_id] = values[level][parent]
            parent = parents[level][parent]
    return np.array(res).astype(values.dtype)

def gather_tree(data):
    num_beams=int(data.shape[-1].value/2)
    values=data[:,:num_beams]
    parents=data[:,num_beams:]
    """Tensor version of gather_tree_py"""
    res = tf.py_func(
        func=gather_tree_py, inp=[values, parents], Tout=values.dtype)
    res.set_shape(values.get_shape().as_list())
    return res

def fix_beam_search_bug(beam_search_outputs):
    '''tensorflow r1.6中BeamSearchDecoder有bug，输出的序列中有多个结束标志<eos>相连，并有重复的beam结果
    按照如下代码进行修复：
    outputs, _,_ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                maximum_iterations=max_length,)
    #predicted_ids=outputs.predicted_ids
    predicted_ids=fix_beam_search_bug(outputs.beam_search_outputs)
    
    '''
    last_output=beam_search_outputs.beam_search_decoder_output
    data=tf.concat([last_output.predicted_ids,last_output.parent_ids],axis=-1)
    predicted_ids=tf.map_fn(gather_tree,data,infer_shape=False)
    return predicted_ids


from tensorflow.python.layers import base
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import nn
from tensorflow.python.eager import context
class OutputWrapper(base.Layer):
    def __init__(self,dense,unknown_id=0,infer=False,name=None,**kwargs):
        '''
        用来防止模型infer的时候经常输出<unk>
        dense: Dense层
        unknown_id: 未知词的id号
        infer：是训练阶段还是推断阶段
        name: 当前层的名称
        '''
        super(OutputWrapper,self).__init__(trainable=True,name=name,**kwargs)
        self.dense=dense
        self.units=self.dense.units
        self.unknown_id=unknown_id
        self.infer=infer
        
    def build(self,input_shape):
        self.dense.build(input_shape)
        
        if self.infer:
            kernel_mask=np.ones(shape=(1000,40004),dtype=np.float32)
            kernel_mask[:,self.unknown_id]=0
            bias_mask=np.ones(shape=(40004,),dtype=np.float32)
            bias_mask[self.unknown_id]=0
            kernel_mask_tensor=tf.constant(kernel_mask,dtype=tf.float32)
            bias_mask_tensor=tf.constant(bias_mask,dtype=tf.float32)
            self.kernel = self.dense.kernel*kernel_mask_tensor
            self.bias = self.dense.bias*bias_mask_tensor
        else:
            self.kernel=self.dense.kernel
            self.bias=self.dense.bias
            
        self.built=True
        
    def call(self,inputs):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        shape = inputs.get_shape().as_list()

        if len(shape) > 2:
          # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, self.kernel, [[len(shape) - 1],[0]])
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
    
    

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

from tensorflow.python.ops.rnn_cell_impl import _Linear,RNNCell,LSTMStateTuple
class CustomLSTMCell(RNNCell):

    def __init__(self, num_units,
               use_peepholes=False, cell_clip=None,
               initializer=None, num_proj=None, proj_clip=None,
               num_unit_shards=None, num_proj_shards=None,
               forget_bias=1.0, state_is_tuple=True,
               activation=None, reuse=None):

        super(CustomLSTMCell, self).__init__(_reuse=reuse)
        

        self._num_units = num_units
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._initializer = initializer
        self._num_proj = num_proj
        self._proj_clip = proj_clip
        self._num_unit_shards = num_unit_shards
        self._num_proj_shards = num_proj_shards
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh

        if num_proj:
            self._state_size = (
              LSTMStateTuple(num_units, num_proj)
              if state_is_tuple else num_units + num_proj)
            self._output_size = num_proj
        else:
            self._state_size = (
              LSTMStateTuple(num_units, num_units)
              if state_is_tuple else 2 * num_units)
            self._output_size = num_units
        self._linear1 = None
        self._linear2 = None
        if self._use_peepholes:
            self._w_f_diag = None
            self._w_i_diag = None
            self._w_o_diag = None

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, state):

        num_proj = self._num_units if self._num_proj is None else self._num_proj
        sigmoid = math_ops.sigmoid

        (c_prev, m_prev) = state

        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
        if self._linear1 is None:
            scope = vs.get_variable_scope()
            with vs.variable_scope(
                scope, initializer=self._initializer) as unit_scope:
                if self._num_unit_shards is not None:
                    unit_scope.set_partitioner(
                      partitioned_variables.fixed_size_partitioner(
                          self._num_unit_shards))
                self._linear1 = _Linear([inputs, m_prev], 4 * self._num_units, True)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        lstm_matrix = self._linear1([inputs, m_prev])
        
        
        i, j, f, o = array_ops.split(
            value=lstm_matrix, num_or_size_splits=4, axis=1)
        # Diagonal connections
        if self._use_peepholes and not self._w_f_diag:
            scope = vs.get_variable_scope()
            with vs.variable_scope(scope, initializer=self._initializer) as unit_scope:
                with vs.variable_scope(unit_scope):
                    self._w_f_diag = vs.get_variable(
                      "w_f_diag", shape=[self._num_units], dtype=dtype)
                    self._w_i_diag = vs.get_variable(
                      "w_i_diag", shape=[self._num_units], dtype=dtype)
                    self._w_o_diag = vs.get_variable(
                      "w_o_diag", shape=[self._num_units], dtype=dtype)

        if self._use_peepholes:
            c = (sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * c_prev +
               sigmoid(i + self._w_i_diag * c_prev) * self._activation(j))
        else:
            c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
               self._activation(j))

        if self._cell_clip is not None:
          # pylint: disable=invalid-unary-operand-type
            c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
          # pylint: enable=invalid-unary-operand-type
        if self._use_peepholes:
            m = sigmoid(o + self._w_o_diag * c) * self._activation(c)
        else:
            m = sigmoid(o) * self._activation(c)

        if self._num_proj is not None:
            if self._linear2 is None:
                scope = vs.get_variable_scope()
                with vs.variable_scope(scope, initializer=self._initializer):
                      with vs.variable_scope("projection") as proj_scope:
                        if self._num_proj_shards is not None:
                              proj_scope.set_partitioner(
                                  partitioned_variables.fixed_size_partitioner(
                                      self._num_proj_shards))
                        self._linear2 = _Linear(m, self._num_proj, False)
            m = self._linear2(m)

            if self._proj_clip is not None:
                # pylint: disable=invalid-unary-operand-type
                m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
                # pylint: enable=invalid-unary-operand-type

        new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                     array_ops.concat([c, m], 1))
        return m, new_state
    
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

class RolloutEmbeddingHelper(Helper):
    def __init__(self, targets, sequence_length, embedding, start_tokens, end_token, start_position, softmax_temperature=None,
                 seed=None,name=None):

        if callable(embedding):
            self._embedding_fn = embedding
        else:
            self._embedding_fn = (
              lambda ids: embedding_ops.embedding_lookup(embedding, ids))

        self._start_tokens = ops.convert_to_tensor(
            start_tokens, dtype=dtypes.int32, name="start_tokens")
        
        self._end_token = ops.convert_to_tensor(
            end_token, dtype=dtypes.int32, name="end_token")
        
        self._start_position=ops.convert_to_tensor(start_position,dtype=dtypes.int32,name='start_position')
        
        if self._start_tokens.get_shape().ndims != 1:
            raise ValueError("start_tokens must be a vector")
        self._batch_size = array_ops.size(start_tokens)
        if self._end_token.get_shape().ndims != 0:
            raise ValueError("end_token must be a scalar")
            
        self._start_inputs = self._embedding_fn(self._start_tokens)

        with ops.name_scope(name, "RolloutEmbeddingHelper", [targets, sequence_length]):
            targets = ops.convert_to_tensor(targets, name="targets")  
            
            targets = nest.map_structure(_transpose_batch_time, targets)   # time x batch x emb

            self._target_tas = nest.map_structure(_unstack_ta, targets)
            
            self._sequence_length = ops.convert_to_tensor(
              sequence_length, name="sequence_length")

        self._softmax_temperature = softmax_temperature
        self._seed = seed
            
    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tensor_shape.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return dtypes.int32

    def initialize(self, name=None):
        finished = array_ops.tile([False], [self._batch_size])
        return (finished, self._start_inputs)

    def sample(self, time, outputs, state, name=None):
        start_sample=(time>=self._start_position)  #开始输出，之前则直接把targets输出
        
        if self._softmax_temperature is None:
            logits = outputs
        else:
            logits = outputs / self._softmax_temperature

        sample_id_sampler = categorical.Categorical(logits=logits)
        sample_ids = sample_id_sampler.sample(seed=self._seed)
        
        def read_from_ta(inp):
            return inp.read(time)
        
        sample_ids=control_flow_ops.cond(start_sample,
                             lambda:sample_ids,
                             lambda:nest.map_structure(read_from_ta,self._target_tas))
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        del time, outputs  # unused by next_inputs_fn
        finished = math_ops.equal(sample_ids, self._end_token)
        all_finished = math_ops.reduce_all(finished)
        next_inputs = control_flow_ops.cond(
            all_finished,
            # If we're finished, the next_inputs value doesn't matter
            lambda: self._start_inputs,
            lambda: self._embedding_fn(sample_ids))
        return (finished, next_inputs, state)

    
'''
加密LSTM代码，tensorflow1.4
'''
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
            self._linear = _Linear([inputs, h, e], 4 * self._num_units, True)
        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = array_ops.split(
            value=self._linear([inputs, h, e]), num_or_size_splits=4, axis=1)

        new_c = (
            c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
        new_h = self._activation(new_c) * sigmoid(o)

        new_state = EncryptLSTMStateTuple(new_c, new_h, e)
        
        return new_h, new_state
    
    def initial_state(self,batch_size,dtype,encrypt_state):
        state=self.zero_state(batch_size,dtype)
        return EncryptLSTMStateTuple(state.c,state.h,encrypt_state)
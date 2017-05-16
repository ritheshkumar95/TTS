# -*- coding: UTF-8 -*-
import lib
import numpy as np
import numpy
import theano
import theano.tensor as T
theano.config.floatX='float32'
#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams
import time
import lasagne
import math

srng = RandomStreams(seed=234)

def initializer(name,shape,val=0,gain='relu',std=None,mean=0.0,range=0.01,alpha=0.01):
    if gain in ['linear','sigmoid']:
        gain = 1.0
    elif gain=='leakyrelu':
        gain = np.sqrt(2/(1+alpha**2))
    elif gain=='relu':
        gain = np.sqrt(2)
    else:
        raise NotImplementedError

    if name=='Constant':
        return lasagne.init.Constant(val).sample(shape)
    elif name=='Normal':
        return lasagne.init.Normal(std if std!=None else 0.01,mean).sample(shape)
    elif name=='Uniform':
        return lasagne.init.Uniform(range=range,std=std,mean=mean).sample(shape)
    elif name=='GlorotNormal':
        return lasagne.init.GlorotNormal(gain=gain).sample(shape)
    elif name=='GlorotUniform':
        return lasagne.init.GlorotUniform(gain=gain).sample(shape)
    elif name=='HeNormal':
        return lasagne.init.HeNormal(gain=gain).sample(shape)
    elif name=='HeUniform':
        return lasagne.init.HeUniform(gain=gain).sample(shape)
    elif name=='Orthogonal':
        return lasagne.init.Orthogonal(gain=gain).sample(shape)
    else:
        return lasagne.init.GlorotUniform(gain=gain).sample(shape)

def Batchnorm(
    name,
    inputs,
    input_dim,
    axes=None,
    mode='train',
    trainable_weights=True
):
    #mult = lib.floatX(0.1) if trainable_weights else lib.floatX(1)
    gamma = lib.param(
        name+'.gamma',
        initializer('Normal',(input_dim,),mean=1.0,std=0.02),
        is_param=trainable_weights
    )
    beta = lib.param(
        name+'.beta',
        np.zeros(input_dim).astype(theano.config.floatX),
        is_param=trainable_weights
    )
    running_mean = lib.param(
        name+'.running_mean',
        np.zeros(input_dim).astype(theano.config.floatX),
        is_param=False
    )
    running_var = lib.param(
        name+'.running_variance',
        np.zeros(input_dim).astype(theano.config.floatX),
        is_param=False
    )

    if mode=='train':
        out,_,_,new_mean,new_var = T.nnet.bn.batch_normalization_train(
            inputs,
            axes=axes,
            gamma=gamma,
            beta=beta,
            running_mean=running_mean,
            running_var=running_var
        )
        lib._updates[running_mean] = new_mean
        lib._updates[running_var] = new_var
        return out

    elif mode=='test':
        return T.nnet.bn.batch_normalization_test(
            inputs,
            axes=axes,
            gamma=gamma,
            beta=beta,
            mean=running_mean,
            var=running_var
        )

def get_fans(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out

def conv2d(
    name,
    input,
    kernel,
    stride,
    depth,
    n_filters,
    init=None,
    bias=True,
    batchnorm=False,
    train_bn=True,
    weightnorm=True,
    pad='valid',
    filter_dilation=(1,1),
    mode='train',
    **kwargs
    ):
    if isinstance(kernel, int):
        kernel_h = kernel_w = kernel
    else:
        kernel_h, kernel_w = kernel

    filter_values = initializer(init,(n_filters,depth,kernel_h,kernel_w),**kwargs)
    #weight_values = lasagne.init.HeNormal().sample((n_filters,depth,kernel_h,kernel_w)).astype('float32')

    W = lib.param(
        name+'.W',
        filter_values
        )

    if weightnorm:
        norm_values = np.linalg.norm(filter_values.reshape((filter_values.shape[0], -1)), axis=1)
        norms = lib.param(
            name + '.g',
            norm_values
        )
        W = W * (norms / W.reshape((W.shape[0],-1)).norm(2, axis=1)).dimshuffle(0,'x','x','x')

    out = T.nnet.conv2d(input,W,subsample=(stride,stride),border_mode=pad,filter_dilation=filter_dilation)

    if bias:
        b = lib.param(
            name + '.b',
            np.zeros(n_filters).astype('float32')
            )

        out += b[None,:,None,None]

    if batchnorm:
        out = Batchnorm(name,out,n_filters,axes='spatial',mode=mode,trainable_weights=train_bn)

    return out

def max_pool(name,input,k,s,pad=(0,0),mode='max'):
    import theano.tensor.signal.pool
    if type(k)==int:
        k = (k,k)
    if type(s)==int:
        s = (s,s)
    return T.signal.pool.pool_2d(input,k,st=s,padding=pad,mode=mode)

def dropout(X, p=0.):
    retain_prob = 1 - p
    return theano.ifelse.ifelse(
        T.gt(p,0),
        X*srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)/retain_prob,
        X
        )

def Linear(
    name,
    inputs,
    input_dim,
    output_dim,
    bias=True,
    init=None,
    weightnorm=True,
    batchnorm=False,
    train_bn=True,
    mode='train',
    **kwargs
    ):

    weight_values = initializer(init,(input_dim,output_dim),**kwargs)
    weight = lib.param(
        name + '.W',
        weight_values
    )

    if weightnorm:
        norm_values = numpy.linalg.norm(weight_values, axis=0)
        norms = lib.param(
            name + '.g',
            norm_values
        )

        prepared_weight = weight * (norms / weight.norm(2, axis=0)).dimshuffle('x', 0)
    else:
        prepared_weight = weight

    if inputs.ndim==3:
        batch_size = inputs.shape[0]
        hidden_dim = inputs.shape[1]
        reshaped_inputs = inputs.reshape((-1,input_dim))
    else:
        reshaped_inputs = inputs

    result = T.dot(reshaped_inputs, prepared_weight)

    if bias:
        b = lib.param(
            name + '.b',
            numpy.zeros((output_dim,), dtype=theano.config.floatX)
        )
        result += b

    if batchnorm:
        result = Batchnorm(
            name,
           result,
           output_dim,
           axes='per-activation',
           mode=mode,
           trainable_weights=train_bn
           )

    if inputs.ndim==3:
        result = result.reshape((batch_size,hidden_dim,output_dim))

    return result

def Embedding(name, n_symbols, output_dim, indices):
    vectors = lib.param(
        name,
        initializer('Normal', (n_symbols,output_dim), std=1/np.sqrt(output_dim)).astype(theano.config.floatX)
    )

    output_shape = tuple(list(indices.shape) + [output_dim])

    return vectors[indices.flatten()].reshape(output_shape)

def softmax_and_sample(logits):
    old_shape = logits.shape
    flattened_logits = logits.reshape((-1, logits.shape[logits.ndim-1]))
    samples = T.cast(
        srng.multinomial(pvals=T.nnet.softmax(flattened_logits)),
        theano.config.floatX
    ).reshape(old_shape)
    return T.argmax(samples, axis=samples.ndim-1)

def LSTMStep(name, input_dim, hidden_dim, mask_t, x_t, prev_state, use_prev_cell=False, weightnorm=True):
    mask_t = T.shape_padright(mask_t)
    h_tm1 = prev_state[:,:hidden_dim]
    c_tm1 = prev_state[:,hidden_dim:]

    gates = lib.ops.Linear(
            name+'.Recurrent_Gates',
            T.concatenate([x_t,h_tm1], 1),
            input_dim+hidden_dim,
            4 * hidden_dim,
            weightnorm=weightnorm
        )

    if use_prev_cell:
        c_tm1_to_gates = lib.ops.Linear(
            name + '.c_tm1_to_gates',
            c_tm1,
            hidden_dim,
            2*hidden_dim,
            bias=False,
            weightnorm=weightnorm
        )

    if not use_prev_cell:
        pre_gates = T.nnet.sigmoid(gates[:,:3*hidden_dim])
        i_t = pre_gates[:,:hidden_dim]
        f_t = pre_gates[:,hidden_dim:2*hidden_dim]
        o_t = pre_gates[:,2*hidden_dim:]
        g_t = T.tanh(gates[:,3*hidden_dim:])

        c_t = f_t*c_tm1 + i_t*g_t
        h_t = o_t*T.tanh(c_t)

    else:
        i_t = T.nnet.sigmoid(gates[:,:hidden_dim] + c_tm1_to_gates[:,:hidden_dim])
        f_t = T.nnet.sigmoid(gates[:,hidden_dim:2*hidden_dim] + c_tm1_to_gates[:,hidden_dim:])
        c_t = f_t*c_tm1 + i_t*T.tanh(gates[:,2*hidden_dim:3*hidden_dim])

        cell_to_output = lib.ops.Linear(
            name + '.c_t_to_outputgate',
            c_t,
            hidden_dim,
            hidden_dim,
            bias=False,
            weightnorm=weightnorm
        )

        o_t = T.nnet.sigmoid(gates[:,3*hidden_dim:]+cell_to_output)
        h_t = o_t*T.tanh(c_t)

    output = T.concatenate([h_t,c_t],1)
    output = mask_t*output + (1-mask_t)*prev_state
    return output

def GRUStep(name, input_dim, hidden_dim, mask_t, x_t, h_tm1, weightnorm=True):
    mask_t = T.shape_padright(mask_t)
    gates = T.nnet.sigmoid(
        lib.ops.Linear(
            name+'.Gates',
            T.concatenate([x_t, h_tm1],1),
            input_dim+hidden_dim,
            2 * hidden_dim,
            weightnorm=weightnorm
        )
    )

    update = gates[:,:hidden_dim]
    reset = gates[:,hidden_dim:]
    scaled_state = reset * h_tm1

    candidate = T.tanh(
        lib.ops.Linear(
            name+'.Candidate',
            T.concatenate([x_t, scaled_state],1),
            input_dim+hidden_dim,
            hidden_dim,
            weightnorm=weightnorm
        )
    )

    output = (update * candidate) + ((1 - update) * h_tm1)
    output = mask_t*output + (1-mask_t)*h_tm1

    return output

def StackedRNNStep(type, name, input_dim, hidden_dim, mask_t, x_t, prev_state, weightnorm=True, n_layers=1, residual=False):
    RNNFunc = LSTMStep if type=='LSTM' else GRUStep
    output_arr = []
    input_to_gru = x_t
    for i in xrange(n_layers):
        state_tm1 = prev_state[:,i]
        state_t = RNNFunc(name+'.Layer%d'%(i+1), input_dim if i==0 else hidden_dim, hidden_dim, mask_t, input_to_gru, state_tm1, weightnorm=weightnorm)
        output_arr += [state_t]
        # If LSTM need to take only h_t not c_t
        h_t = state_t[:,:hidden_dim]
        if residual:
            input_to_gru += h_t
        else:
            input_to_gru = h_t

    return T.patternbroadcast(T.stack(output_arr,axis=1),(False,False,False))

def StackedOpenLoopRNNStep(type, name, input_dim, hidden_dim, output_dim, mask_t, x_t, prev_output, prev_state, weightnorm=True, n_layers=1, residual=False, speaker=True):
    input = T.concatenate([prev_output,x_t],-1)
    state_t = StackedRNNStep(type, name, input_dim+output_dim, hidden_dim, mask_t, input, prev_state, weightnorm=weightnorm, n_layers=n_layers, residual=residual)
    if speaker:
        readout = lib.ops.Linear(
            name+'.Output.MLP.1',
            T.concatenate([state_t[:,-1,:hidden_dim],x_t[:,-128:]],-1),
            hidden_dim+128,
            output_dim,
            weightnorm=weightnorm
        )
    else:
        readout = lib.ops.Linear(
            name+'.Output.MLP.1',
            state_t[:,-1,:hidden_dim],
            hidden_dim,
            output_dim,
            weightnorm=weightnorm
        )
    return readout,state_t

def OpenLoopDecoderStep(type, name, input_dim, hidden_dim, output_dim, mask_t, prev_output, prev_state, weightnorm=True, n_layers=1, residual=False):
    x_t = lib.ops.Embedding(
        'NMT.Embedding_Phons',
        45,
        input_dim,
        prev_output
    )
    state_t = StackedRNNStep(type, name, input_dim, hidden_dim, mask_t, x_t, prev_state, weightnorm=weightnorm, n_layers=n_layers, residual=residual)
    logits = T.nnet.softmax(lib.ops.Linear(
        name+'.Output.MLP.1',
        state_t[:,-1,:hidden_dim],
        hidden_dim,
        45
    ))
    idxs = T.argmax(logits,axis=-1).astype('int32')
    return idxs,state_t

def RNN(type, name, inputs, input_dim, hidden_dim, output_dim=None, mode='train',h0=None, mask=None, backward=False, weightnorm=True, n_layers=1, residual=False, **kwargs):
    #inputs.shape = (batch_size,N_FRAMES,FRAME_SIZE)
    size = 2*hidden_dim if type=='LSTM' else hidden_dim
    if inputs is not None:
        batch_size = T.shape(inputs)[0]
    elif h0 is not None:
        batch_size = T.shape(h0)[0]
    else:
        batch_size = kwargs['batch_size']

    if inputs is not None:
        seq_len = T.shape(inputs)[1]
    else:
        try:
            seq_len = kwargs['seq_len']
        except:
            print "Missing seq_len argument! Using 200"
            seq_len = 200

    if h0 is None:
        h0 = T.tile(lib.param(
            name+'.h0',
            np.zeros((1,n_layers,size)).astype('float32')
        ), [batch_size,1,1])

    if mask is None:
        mask = T.ones((seq_len,batch_size),dtype=theano.config.floatX)
    else:
        mask = mask.dimshuffle(1,0)

    def train_step(mask_t, x_t, state_tm1):
        return StackedRNNStep(
            type,
            name,
            input_dim,
            hidden_dim,
            mask_t,
            x_t,
            state_tm1,
            weightnorm=weightnorm,
            n_layers=n_layers,
            residual=residual
        )

    def open_loop_step(mask_t, x_t, prev_output, state_tm1):
        return StackedOpenLoopRNNStep(
            type,
            name,
            input_dim,
            hidden_dim,
            output_dim,
            mask_t,
            x_t,
            prev_output,
            state_tm1,
            weightnorm=weightnorm,
            n_layers=n_layers,
            residual=residual,
            **kwargs
        )

    def decoder_step(mask_t, prev_output, state_tm1):
        return OpenLoopDecoderStep(
            type,
            name,
            input_dim,
            hidden_dim,
            output_dim,
            mask_t,
            prev_output,
            state_tm1,
            weightnorm=weightnorm,
            n_layers=n_layers,
            residual=residual
        )

    if mode=='train':
        inputs = inputs.dimshuffle(1,0,2)
        states, _ = theano.scan(
            train_step,
            sequences=[mask,inputs],
            outputs_info=[h0],
            go_backwards=backward
        )

        states = states.dimshuffle(1,0,2,3)
        #if LSTM return only h_t not c_t
        if 'return_cell' in kwargs and kwargs['return_cell']=True:
            return states
        else:
            return states[:,:,:,:hidden_dim]

    elif mode=='open-loop-rnn':
        inputs = inputs.dimshuffle(1,0,2)
        (outputs,states), _ = theano.scan(
            open_loop_step,
            sequences=[mask,inputs],
            outputs_info=[T.alloc(0.,batch_size,output_dim).astype(theano.config.floatX),h0],
            go_backwards=backward
        )
        return outputs.dimshuffle(1,0,2)

    else:
        n_input = kwargs['n_input']
        (cap,outputs), _ = theano.scan(
            decoder_step,
            sequences=[mask],
            outputs_info=[T.repeat(T.as_tensor_variable(n_input-1),batch_size).astype('int32'),h0],
            go_backwards=backward,
            n_steps=seq_len
        )
        return cap.T

def BiRNN(type, name, inputs, input_dim, hidden_dim, mask=None, weightnorm=True, n_layers=1, residual=False):
    h_t1 = RNN(type, name+'.Forward', inputs, input_dim, hidden_dim, mask=mask, weightnorm=weightnorm, n_layers=n_layers, residual=residual)
    h_t2 = RNN(type, name+'.Backward', inputs, input_dim, hidden_dim, mask=mask, backward=True, weightnorm=weightnorm, n_layers=n_layers, residual=residual)
    return T.concatenate([h_t1,h_t2],-1)

def AttnDecStep(name, n_input, input_dim, hidden_dim, ctx_dim, ctx, x_t, prev_state, mode='train',weightnorm=True):
    # h_tm1 = prev_state[:,:hidden_dim]
    # c_tm1 = prev_state[:,hidden_dim:]
    h_tm1 = prev_state
    if mode=='open-loop':
        x_t = lib.ops.Embedding(
            'NMT.Embedding_Phons',
            n_input,
            input_dim,
            x_t
            )

    tiled_h_tm1 = T.tile(h_tm1[:,None,:],[1,ctx.shape[1],1])
    e_vec = T.nnet.relu(lib.ops.Linear(
        'NMT.Attention.MLP1',
        T.concatenate([tiled_h_tm1,ctx],-1),
        hidden_dim+ctx_dim,
        hidden_dim
    ))
    e_vec = T.nnet.softmax(lib.ops.Linear(
        'NMT.Attention.MLP2',
        e_vec,
        hidden_dim,
        1
    )[:,:,0]) # (B, seq_len)

    c_t = T.sum(T.shape_padright(e_vec)*ctx,axis=1)
    input_to_rnn = T.concatenate([x_t,c_t],-1)
    # state_t = LSTMStep(name,False, input_dim+ctx_dim, hidden_dim, input_to_rnn, prev_state)
    mask_t = T.ones((x_t.shape[0],)).astype(theano.config.floatX)
    state_t = GRUStep(name, input_dim+ctx_dim, hidden_dim, mask_t, input_to_rnn, h_tm1)
    if mode=='open-loop':
        logits = T.nnet.softmax(lib.ops.Linear(
            name+'.Output.MLP.1',
            T.concatenate([x_t,state_t[:,:hidden_dim],c_t],-1),
            input_dim+hidden_dim+ctx_dim,
            n_input
        ))
        idxs = T.argmax(logits,axis=-1).astype('int32')
        return idxs,state_t
    else:
        return state_t,c_t

def AttnDec(name, context, input_dim, n_input, context_dim, hidden_dim, inputs=None, mode='train',h0=None, backward=False, weightnorm=True):
    #inputs.shape = (batch_size,N_FRAMES,FRAME_SIZE)
    def step(x_t, h_tm1):
        return AttnDecStep(
            name,
            n_input,
            input_dim,
            hidden_dim,
            context_dim,
            context,
            x_t,
            h_tm1,
            mode=mode,
            weightnorm=weightnorm
        )

    if h0 is None:
        # size = 2*hidden_dim
        size = hidden_dim
        batch_size=context.shape[0]
        h0 = T.tile(lib.param(
            name+'.h0',
            np.zeros((1,size)).astype('float32')
            ), [batch_size,1])

    if mode=='train':
        (output1,output2), _ = theano.scan(
            step,
            sequences=[inputs.transpose(1,0,2)],
            outputs_info=[h0,None],
            go_backwards=backward
        )
    else:
        batch_size = T.shape(h0)[0]
        (cap,outputs), _ = theano.scan(
            step,
            sequences=None,
            outputs_info=[T.repeat(T.as_tensor_variable(n_input-1),batch_size).astype('int32'),h0],
            go_backwards=backward,
            n_steps=200
        )
        return cap.T

    out1 = output1.dimshuffle(1,0,2)
    out2 = output2.dimshuffle(1,0,2)
    return out1,out2

def TLAttnDecStep(name, input_dim, n_input, hidden_dim, ctx_dim, ctx, x_t, h_tm1, type='LSTM', mode='train',weightnorm=True):
    state_end_idx = 2*hidden_dim if type=='LSTM' else 1*hidden_dim
    state = h_tm1[:,:state_end_idx]
    feed = h_tm1[:,state_end_idx:]
    if mode=='open-loop':
        x_t = lib.ops.Embedding(
            'NMT.Embedding_Phons',
            n_input,
            input_dim,
            x_t
            )
        print "going in"
    input_to_rnn = T.concatenate([x_t,feed],-1)
    mask_t = T.ones((x_t.shape[0],)).astype(theano.config.floatX)
    if type=='LSTM':
        state_t = LSTMStep(name, input_dim+hidden_dim, hidden_dim, mask_t, input_to_rnn, state)
        h_t = state_t[:,:hidden_dim]
        c_t = state_t[:,hidden_dim:]
    elif type=='GRU':
        state_t = GRUStep(name, input_dim+hidden_dim, hidden_dim, mask_t, input_to_rnn, state)
        h_t = state_t

    target_t = T.shape_padright(lib.ops.Linear(
        name+'.target_t',
        h_t,
        hidden_dim,
        ctx_dim,
        bias=False
    ))

    a_t = T.nnet.softmax(T.batched_dot(ctx,target_t)[:,:,0])
    z_t = T.sum(T.shape_padright(a_t)*ctx,axis=1)

    output_t = T.tanh(lib.ops.Linear(
        name+'.output_t',
        T.concatenate([h_t,z_t],-1),
        ctx_dim+hidden_dim,
        hidden_dim,
        bias=False
        ))

    new_state = T.concatenate([state_t,output_t],-1)
    if mode=='open-loop':
        logits = T.nnet.softmax(lib.ops.Linear(
            name+'.Output.MLP.1',
            output_t,
            hidden_dim,
            n_input
        ))
        idxs = T.argmax(logits,axis=-1).astype('int32')
        return idxs,new_state
    else:
        return new_state

def TLAttnDec(name, type, context, input_dim, n_input, context_dim, hidden_dim, inputs=None, mode='train',h0=None, backward=False, weightnorm=True):
    #inputs.shape = (batch_size,N_FRAMES,FRAME_SIZE)
    def step(x_t, h_tm1):
        return TLAttnDecStep(
            name,
            input_dim,
            n_input,
            hidden_dim,
            context_dim,
            context,
            x_t,
            h_tm1,
            type=type,
            mode=mode,
            weightnorm=weightnorm
        )

    if h0 is None:
        size = 3*hidden_dim if type=='LSTM' else 2*hidden_dim
        batch_size=context.shape[0]
        h0 = T.tile(lib.param(
            name+'.h0',
            np.zeros((1,size)).astype('float32')
            ), [batch_size,1])

    if mode=='train':
        outputs, _ = theano.scan(
            step,
            sequences=[inputs.transpose(1,0,2)],
            outputs_info=[h0],
            go_backwards=backward
        )
    else:
        batch_size = T.shape(h0)[0]
        (cap,outputs), _ = theano.scan(
            step,
            sequences=None,
            outputs_info=[T.repeat(T.as_tensor_variable(n_input-1),batch_size).astype('int32'),h0],
            go_backwards=backward,
            n_steps=100
        )
        return cap.T

    out = outputs.dimshuffle(1,0,2)
    return out

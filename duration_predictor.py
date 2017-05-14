import theano
import theano.tensor as T
import lib
import lib.ops
import numpy as np
import time
import lasagne
import vctk_loader
from datasets import parrot_stream

BATCH_SIZE = 16
V = 43 + 1
N_SPEAKERS = 21+1
NB_EPOCHS = 50
LR = 0.0001
N_LAYERS = 4
GRAD_CLIP = 1
SAVE_FILE_NAME = 'vctk_duration_predictor_best.pkl'

X = T.imatrix()
Y = T.fmatrix()
mask = T.fmatrix()
learn_rate = T.fscalar()
drop_prob = T.fscalar()

def maxout(X):
    if X.ndim==4:
        dim = 1
    else:
        dim = -1
    split_size = X.shape[dim]//2
    split1,split2 = T.split(X,[split_size,split_size],2,axis=dim)
    return T.maximum(split1,split2)

def DurationPredictor(X,drop_prob):
    batch_size = T.shape(X)[0]
    emb_phons = lib.ops.dropout(lib.ops.Embedding(
        'DurationPredictor.Embedding_Phonemes',
        V,
        256,
        X
    ),drop_prob).transpose(0,2,1)[:,:,None,:]

    out = lib.ops.dropout(maxout(lib.ops.conv2d(
        'DurationPredictor.Conv1',
        emb_phons,
        depth=256,
        n_filters=1024,
        kernel=(1,5),
        stride=1,
        pad='half'
    )),drop_prob)

    out = lib.ops.dropout(maxout(lib.ops.conv2d(
        'DurationPredictor.Conv2',
        out,
        depth=512,
        n_filters=2048,
        kernel=(1,5),
        stride=1,
        pad='half'
    )),drop_prob)

    out = lib.ops.dropout(maxout(lib.ops.conv2d(
        'DurationPredictor.Conv3',
        out,
        depth=1024,
        n_filters=1024,
        kernel=(1,5),
        stride=1,
        pad='half'
    )),drop_prob)

    out = T.nnet.relu(lib.ops.conv2d(
        'DurationPredictor.Conv4',
        out,
        depth=512,
        n_filters=1,
        kernel=(1,1),
        stride=1,
        pad='half'
    ))[:,0,0,:]

    return out

def RecurrentPredictor(X,drop_prob,mask=None):
    batch_size = T.shape(X)[0]
    seq_len = T.shape(X)[1]
    emb_phons = lib.ops.dropout(lib.ops.Embedding(
        'DurationPredictor.Embedding_Phonemes',
        V,
        256,
        X
    ),drop_prob)

    gru = lib.ops.dropout(lib.ops.BiGRU(
        'DurationPredictor.BiGRU',
        emb_phons,
        256,
        256,
        mask=mask
    ),drop_prob)

    out = T.nnet.relu(lib.ops.Linear(
        'DurationPredictor.FC',
        gru,
        512,
        1
    ))[:,:,0]

    return out

def DeepVoice(X, drop_prob):
    batch_size = T.shape(X)[0]
    seq_len = T.shape(X)[1]
    emb_phons = T.extra_ops.to_one_hot(X.flatten(),V).reshape((batch_size,-1,V))
    out = T.nnet.relu(lib.ops.Linear(
        'DurationPredictor.FC.1',
        emb_phons,
        V,
        256
    ))

    out = lib.ops.dropout(T.nnet.relu(lib.ops.Linear(
        'DurationPredictor.FC.2',
        out,
        256,
        256
    )),drop_prob)

    out = lib.ops.dropout(lib.ops.RNN(
        'GRU',
        'DurationPredictor.GRU',
        out,
        256,
        128,
        n_layers=2,
        residual=True
    )[:,:,-1],drop_prob)

    out = lib.ops.Linear(
        'DurationPredictor.FC.3',
        out,
        128,
        1
    )[:,:,0]
    return out

def getDurations(train_labs):
    batch_size = train_labs.shape[0]
    from itertools import groupby
    labels = []
    durations = []
    for i in xrange(batch_size):
        groups = [(k,sum(1 for _ in g)) for k, g in groupby(train_labs[i])]
        a,b = zip(*groups)
        labels.append(list(a))
        durations.append(list(b))
        assert len(a)==len(b),"Invalid grouping"
    max_len = max([len(x) for x in labels])
    mask = np.zeros((batch_size,max_len)).astype(np.float32)
    for i in xrange(batch_size):
        mask[i,:len(labels[i])] = 1.
        labels[i] += (0,)*(max_len-len(labels[i]))
        durations[i] += (0,)*(max_len-len(durations[i]))
    return np.asarray(labels,dtype=np.int32),np.asarray(durations,dtype=np.float32),mask

def score(batch_size=16):
    start = time.time()
    valid_costs = []
    # valid_stream = parrot_stream(
    #     'vctk',
    #     True,
    #     ('valid',),
    #     batch_size,
    #     noise_level = 0.,
    #     labels_type='phonemes',
    #     seq_size=100000
    # )
    # itr = valid_stream.get_epoch_iterator()
    # for val_X,val_mask,val_ctx,val_spk,val_reset,val_noise_level in itr:
    #     val_X = val_X.transpose((1,0,2))
    #     val_mask =val_mask.T
    #     val_ctx = val_ctx.T
        # test_X,test_Y,test_mask = getDurations(val_ctx)
    valid_itr = vctk_loader.duration_data_loader('valid',batch_size=BATCH_SIZE)
    for test_X,test_Y,test_mask in valid_itr:
        valid_loss = test_fn(
                0.,
                test_X,
                test_Y.astype('float32'),
                test_mask
            )
        valid_costs.append(valid_loss)
    print "Validation Completed! cost: {} time: {}".format(np.mean(np.asarray(valid_costs),axis=0),time.time()-start)
    return np.mean(np.asarray(valid_costs),axis=0)[1]

if __name__=='__main__':
    batch_size = T.shape(X)[0]
    # readout = DurationPredictor(X,drop_prob)
    # readout = RecurrentPredictor(X,drop_prob,mask=mask)
    readout = DeepVoice(X,drop_prob)
    batch_size = T.shape(readout)[0]
    cost = T.sum(T.sqr(readout-Y)*mask)/T.sum(mask)
    abs_cost = T.sum(T.abs_(readout-Y)*mask)/T.sum(mask)

    params = lib.search(cost, lambda x: hasattr(x, "param"))
    lib.print_params_info(params)
    grads = T.grad(cost, wrt=params, disconnected_inputs='warn')
    grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in grads]

    print "Gradients Computed"
    updates = lasagne.updates.adam(grads, params, learning_rate=learn_rate)

    train_fn = theano.function(
        [drop_prob,X,Y,mask,learn_rate],
        [cost,abs_cost],
        updates=updates,
        on_unused_input='warn'
    )

    test_fn = theano.function(
        [drop_prob,X,Y,mask],
        [cost,abs_cost],
        on_unused_input='warn'
    )

    print "Compiled Functions!"
    i=0
    best_cost = 1000.
    for i in xrange(i, NB_EPOCHS):
        iteration = 0
        costs = []
        times = []
        # train_stream = parrot_stream(
        #     'vctk',
        #     True,
        #     ('train',),
        #     BATCH_SIZE,
        #     noise_level = 0.,
        #     labels_type='phonemes',
        #     seq_size=100000
        # )
        # itr = train_stream.get_epoch_iterator()
        # for train_X,train_mask,train_ctx,train_spk,train_reset,train_noise_level in itr:
        #     train_X = train_X.transpose((1,0,2))
        #     train_mask = train_mask.T
        #     train_ctx = train_ctx.T
        #     train_X,train_Y,train_mask = getDurations(train_ctx)
        itr = vctk_loader.duration_data_loader('train',batch_size=BATCH_SIZE)
        for train_X,train_Y,train_mask in itr:
            start = time.time()
            iteration += 1
            loss = train_fn(
                    .2,
                    train_X,
                    train_Y.astype('float32'),
                    train_mask,
                    LR
                )

            times.append(time.time()-start)
            costs.append(loss)

        print "Epoch {} Completed! cost: {} time: {}" .format(i + 1,np.mean(np.asarray(costs),axis=0),np.mean(times))
        valid_cost = score()
        if valid_cost < best_cost:
            best_cost = valid_cost
            lib.save_params(SAVE_FILE_NAME)

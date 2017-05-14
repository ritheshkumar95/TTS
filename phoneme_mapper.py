import theano.tensor as T
import theano
import lib
import lib.ops
import numpy as np
import time
from datasets import parrot_stream
import lasagne
from recognizer import Recognizer

BATCH_SIZE      = 16
EMB_DIM         = 256
SPEAKER_DIM     = 128
ENC_DIM         = 128
V               = 43+1
NB_EPOCHS       = 40
N_SPEAKERS      = 21+1
OUTPUT_DIM      = 63
LR              = 0.001
SAVE_FILE_NAME  = 'blizzard_cnn_mapper.pkl'
WEIGHTNORM      = True

X = T.ftensor3()
mask = T.fmatrix()
ctx = T.imatrix()
learn_rate = T.fscalar()

def RecurrentMapper(ctx):
    emb_ctx = lib.ops.Embedding(
        'Mapper.Generator.Embedding_Context',
        V,
        ENC_DIM,
        ctx
    )
    batch_size = T.shape(ctx)[0]
    seq_len = T.shape(ctx)[1]
    out = lib.ops.BiGRU(
        'Mapper.Generator.BiGRU',
        emb_ctx,
        ENC_DIM,
        256
    )
    readout = lib.ops.Linear(
        'Mapper.Generator.FC',
        out,
        512,
        EMB_DIM
    )
    return readout

def ConvolutionalMapper(ctx,mode='train'):
    emb_ctx = lib.ops.Embedding(
        'Mapper.Generator.Embedding_Context',
        V,
        ENC_DIM,
        ctx
    )

    batch_size = T.shape(ctx)[0]
    seq_len = T.shape(ctx)[1]

    # weights_param = {
    #     'init': 'Normal',
    #     'weightnorm': False,
    #     'bias': False,
    #     'std': 0.02
    # }

    out = T.nnet.relu(lib.ops.conv2d(
        'Mapper.Generator.CNN.1',
        emb_ctx.transpose(0,2,1)[:,:,None,:],
        kernel=(1,21),
        stride=1,
        depth=ENC_DIM,
        n_filters=512,
        pad='half'
        # batchnorm=True,
        # mode=mode,
        # **weights_param
    ))

    out = T.nnet.relu(lib.ops.conv2d(
        'Mapper.Generator.CNN.2',
        out,
        kernel=(1,11),
        stride=1,
        depth=512,
        n_filters=512,
        pad='half'
        # batchnorm=True,
        # mode=mode,
        # **weights_param
    ))

    out = T.nnet.relu(lib.ops.conv2d(
        'Mapper.Generator.CNN.3',
        out,
        kernel=(1,5),
        stride=1,
        depth=512,
        n_filters=256,
        pad='half'
        # batchnorm=True,
        # mode=mode,
        # **weights_param
    ))

    '''
    Changing Here! No Activation!!!!!!!
    '''
    readout = lib.ops.conv2d(
        'Mapper.Generator.CNN.4',
        out,
        kernel=(1,1),
        stride=1,
        depth=256,
        n_filters=EMB_DIM,
        pad='half'
        # **weights_param
    )[:,:,0,:].transpose(0,2,1)
    return readout


def score(batch_size=16):
    valid_stream = parrot_stream(
        'blizzard',
        False,
        ('valid',),
        batch_size,
        noise_level = 0.,
        labels_type='phonemes',
        seq_size=100000
    )
    itr = valid_stream.get_epoch_iterator()
    costs = []
    times = []
    for val_X,val_mask,val_ctx,val_reset,val_noise_level in itr:
        val_X = val_X.transpose((1,0,2))
        val_mask =val_mask.T
        val_ctx = val_ctx.T
        start = time.time()
        _loss = test_fn(
            val_X,
            val_ctx,
            val_mask
        )
        costs.append(_loss)
        times.append(time.time()-start)

    print "\n\nValidation Completed!"
    print "\tMean cost: ",np.mean(np.asarray(costs),axis=0)/EMB_DIM
    print "\tMean time: ",np.mean(times)
    return np.mean(np.asarray(costs),axis=0)/EMB_DIM

def save():
    import pickle
    save_dict = {x:y.get_value() for x,y in lib._params.iteritems() if 'Mapper' in x}
    pickle.dump(save_dict,open(SAVE_FILE_NAME,'wb'))

if __name__=='__main__':
    rec = Recognizer(X,0.)
    h = rec['CNN.1']
    # readout = RecurrentMapper(ctx)
    readout = ConvolutionalMapper(ctx)
    # predict_readout = create_graph(X_concat,mode='test')
    predict_readout = readout

    mask_mult = T.shape_padright(mask)
    cost = T.sum(T.sqr(readout-h)*mask_mult)/(T.sum(mask_mult))
    test_cost = T.sum(T.sqr(predict_readout-h)*mask_mult)/(T.sum(mask_mult))

    params = lib.search(cost, lambda x: hasattr(x, "param") and x.param==True and 'Mapper.Generator' in x.name)
    lib.print_params_info(params)
    grads = T.grad(cost, wrt=params, disconnected_inputs='warn')

    print "Gradients Computed"

    updates = lasagne.updates.adam(grads, params, learning_rate=learn_rate)
    # for x,y in lib._updates.iteritems():
    #     if 'Generator' in x.name:
    #         updates[x] = y
    #         print "Adding update: ",x.name

    train_fn = theano.function(
        [X,ctx,mask,learn_rate],
        cost,
        updates=updates,
        on_unused_input='warn'
    )

    test_fn = theano.function(
        [X,ctx,mask],
        test_cost,
        on_unused_input='warn'
    )

    lib.load_params('/data/lisa/exp/kumarrit/ppgn-speech/blizzard_tanh_recognizer_best.pkl')

    print "Compiled Function!"

    i=0
    train_stream = parrot_stream(
        'blizzard',
        False,
        ('train',),
        BATCH_SIZE,
        noise_level = 0.,
        labels_type='phonemes',
        seq_size=100000
    )

    i=0
    best_score = 1000.
    for i in xrange(i,NB_EPOCHS):
        costs=[]
        iter=0
        times=[]
        itr = train_stream.get_epoch_iterator()
        for train_X,train_mask,train_ctx,train_reset,train_noise_level in itr:
            train_X = train_X.transpose((1,0,2))
            train_mask = train_mask.T
            train_ctx = train_ctx.T
            iter += 1

            start = time.time()

            _loss = train_fn(
                train_X,
                train_ctx,
                train_mask,
                LR
                )

            times.append(time.time()-start)
            costs.append(_loss)

            if iter%50==0:
                print "Iter: %d (Epoch %d)"%(iter,i+1)
                print "\tMean cost: ",np.mean(np.asarray(costs),axis=0)/EMB_DIM
                print "\tMean time: ",np.mean(times)

        print "\n\nEpoch %d Completed!"%(i+1)
        print "\tMean cost: ",np.mean(np.asarray(costs),axis=0)/EMB_DIM
        print "\tMean time: ",np.mean(times)
        cur_score = score()
        if cur_score<best_score:
            best_score = cur_score
            print "Saving model!"
            save()

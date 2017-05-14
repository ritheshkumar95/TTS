import theano.tensor as T
import theano
import lib
import lib.ops
import numpy as np
import time
from datasets import parrot_stream
import lasagne
import generate
from phoneme_mapper import ConvolutionalMapper
from duration_predictor import DurationPredictor,RecurrentPredictor
import vctk_loader

BATCH_SIZE      = 16
SPEAKER_DIM     = 128
EMB_DIM         = 256
DEC_DIM         = 512
V               = 43+1
NB_EPOCHS       = 50
N_SPEAKERS      = 21+1
OUTPUT_DIM      = 63
LR              = 0.001
GRAD_CLIP       = 1
N_RNN           = 1
SAVE_FILE_NAME  = 'vctk_synthesizer_GRU_l1.pkl'
ENCODING        = True

X = T.ftensor3()
spkr_ids = T.ivector()
ctx = T.imatrix()
mask = T.fmatrix()
learn_rate = T.fscalar()
noise_vocoder = T.fscalar()

batch_size = T.shape(ctx)[0]
seq_len = T.shape(ctx)[1]

emb_spkr = lib.ops.Embedding(
    'Generator.Speaker_Embedding',
    N_SPEAKERS,
    SPEAKER_DIM,
    spkr_ids
)

tiled_speaker = T.tile(emb_spkr[:,None,:],[1,seq_len,1])
if ENCODING:
    emb_ctx = T.concatenate([T.nnet.relu(ConvolutionalMapper(ctx,mode='train')),tiled_speaker],-1)

input_X = T.concatenate([T.zeros((batch_size,1,OUTPUT_DIM)),X[:,:-1]],axis=1) + lib.ops.srng.normal(T.shape(X),std=noise_vocoder)
X_concat = T.concatenate([input_X,emb_ctx],axis=2)
state = lib.ops.RNN(
    'GRU',
    'Generator.GRU',
    X_concat,
    OUTPUT_DIM+EMB_DIM+SPEAKER_DIM,
    DEC_DIM,
    n_layers=N_RNN
)

predict_readout = lib.ops.RNN(
    'GRU',
    'Generator.GRU',
    emb_ctx,
    EMB_DIM+SPEAKER_DIM,
    DEC_DIM,
    OUTPUT_DIM,
    n_layers=N_RNN,
    mode='test'
)

readout = lib.ops.Linear(
    'Output.MLP.1',
    T.concatenate([state[:,:,-1],tiled_speaker],-1),
    DEC_DIM+SPEAKER_DIM,
    OUTPUT_DIM
)

mask_mult = T.shape_padright(mask)

cost = T.sum(T.sqr(X-readout)*mask_mult)/(T.sum(mask)*63.)
test_cost = T.sum(T.sqr(X-predict_readout)*T.shape_padright(mask))/(T.sum(mask)*63.)

params = lib.search(cost, lambda x: hasattr(x, "param") and x.param==True)
lib.print_params_info(params)

grads = T.grad(cost, wrt=params, disconnected_inputs='warn')
grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in grads]

print "Gradients Computed"

updates = lasagne.updates.adam(grads, params, learning_rate=learn_rate)

train_fn = theano.function(
    [noise_vocoder,X,spkr_ids,ctx,mask,learn_rate],
    cost,
    updates=updates,
    on_unused_input='warn'
)

test_fn = theano.function(
    [X,spkr_ids,ctx,mask],
    test_cost,
    on_unused_input='warn'
)

predict_fn = theano.function(
    [spkr_ids,ctx],
    predict_readout,
    on_unused_input='warn'
)

def writewav():
    data = np.load('test_X.npy')
    out_data = []
    for i in xrange(data.shape[0]):
        for j in xrange(1,data.shape[1]):
            if data[i][j].sum()==0:
                break
        out_data.append(data[i][:j])
    [generate.generate_wav(out_data[i],base='generated_sample_%d'%i,do_post_filtering=False) for i in xrange(data.shape[0])]

    data = np.load('train_X.npy')
    out_data = []
    for i in xrange(data.shape[0]):
        for j in xrange(data.shape[1]):
            if data[i][j].sum()==0:
                break
        out_data.append(data[i][:j])
    print data.shape
    [generate.generate_wav(out_data[i],base='original_sample_%d'%i,do_post_filtering=False) for i in xrange(data.shape[0])]

def predict(batch_size=1):
    test_stream = parrot_stream(
        'vctk',
        True,
        ('valid',),
        batch_size,
        noise_level = 0,
        labels_type='phonemes',
        seq_size=3000
    )
    itr = test_stream.get_epoch_iterator()

    # batch = np.random.choice(28)
    # batch=6
    # for j in range(batch):
    #     test_X,test_mask,test_ctx,test_spk,test_reset,test_noise_level = itr.next()

    for j in xrange(2):
        for test_X,test_mask,test_ctx,test_spk,test_reset,test_noise_level in itr:
            if test_X.shape[0]>400:
                break

    test_X = test_X.transpose((1,0,2))
    test_ctx = test_ctx.T
    test_mask = test_mask.T

    train_X = test_X.copy()
    test_X = predict_fn(
          test_spk[:,0],
          test_ctx,
        )*test_mask[:,:,None]

    print np.sum((train_X[:batch_size]-test_X)**2)/np.sum(test_mask)
    np.save('test_X',test_X)
    np.save('train_X',train_X[:batch_size])

def score(batch_size=16):
    costs = []
    times = []
    itr = vctk_loader.data_loader('valid',batch_size)
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
    for val_spk,val_X,val_X_mask,val_ctx,val_ctx_mask in itr:
        start = time.time()
        _loss = test_fn(
            val_X,
            val_spk,
            val_ctx,
            val_X_mask
        )
        costs.append(_loss)
        times.append(time.time()-start)

    print "\n\nValidation Completed!"
    print "\tMean cost: ",np.mean(np.asarray(costs),axis=0)
    print "\tMean time: ",np.mean(times)
    return np.mean(np.asarray(costs),axis=0)

print "Compiled Function!"

i=0
best_cost = 10.
noise_arr = [4.]*10+np.linspace(4.,1.,NB_EPOCHS-10).tolist() + [1.]*10
for i in xrange(i,NB_EPOCHS):
    iter=0
    costs=[]
    times=[]
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
    itr = vctk_loader.data_loader('train',BATCH_SIZE)
    for train_spk,train_X,train_mask,train_ctx,_ in itr:
        iter += 1
        start = time.time()

        _loss = train_fn(
            noise_arr[i],
            train_X,
            train_spk,
            train_ctx,
            train_mask,
            LR
            )

        times.append(time.time()-start)
        costs.append(_loss)

        if iter%50==0:
            print "Iter: {} (Epoch {}) Cost: {} Time: {}".format(iter,i+1,np.mean(np.asarray(costs),axis=0),np.mean(times))

    print "\n\nEpoch %d Completed!"%(i+1)
    print "\tMean train cost: ",np.mean(np.asarray(costs),axis=0)
    print "\tMean time: ",np.mean(times)
    print ""

    cost = score()
    if cost<best_cost:
        best_cost = cost
        lib.save_params(SAVE_FILE_NAME)
        print "Saving Model {}!\n".format(SAVE_FILE_NAME)

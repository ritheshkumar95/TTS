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
from duration_predictor import DurationPredictor,RecurrentPredictor,DeepVoice
from nmt import test_nmt
from tqdm import tqdm
import pickle
import vctk_loader

BATCH_SIZE      = 16
SPEAKER_DIM     = 128
EMB_DIM         = 256
DEC_DIM         = 512
V               = 43+1
NB_EPOCHS       = 60
N_SPEAKERS      = 21+1
OUTPUT_DIM      = 63
LR              = 0.001
GRAD_CLIP       = 1
N_RNN           = 1
ENCODING        = True

spkr_ids = T.ivector()
ctx = T.imatrix()

chars = T.imatrix()
chars_mask = T.fmatrix()
phons = test_nmt(chars,chars_mask)

batch_size = T.shape(ctx)[0]
preds = DeepVoice(ctx,0.)
preds = T.cast(preds,'int32')
aligned_ctx = T.repeat(ctx.flatten(),preds.flatten()).reshape((batch_size,-1))
seq_len = T.shape(aligned_ctx)[1]
batch_size = T.shape(aligned_ctx)[0]

emb_spkr = lib.ops.Embedding(
    'Generator.Speaker_Embedding',
    N_SPEAKERS,
    SPEAKER_DIM,
    spkr_ids
)

tiled_speaker = T.tile(emb_spkr[:,None,:],[1,seq_len,1])
if ENCODING:
    emb_ctx = T.concatenate([T.nnet.relu(ConvolutionalMapper(aligned_ctx,mode='train')),tiled_speaker],-1)

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

lib.print_params_info(lib._params.values())

predict_fn = theano.function(
    [spkr_ids,ctx],
    predict_readout,
    on_unused_input='warn'
)

direct_fn = theano.function(
    [spkr_ids,aligned_ctx],
    predict_readout,
    on_unused_input='warn'
)

nmt_fn = theano.function(
    [chars,chars_mask],
    phons
)

def save(gen,orig,i=0):
    generate.generate_wav(gen,base='generated_sample_%d'%i,do_post_filtering=False)
    generate.generate_wav(orig,base='original_sample_%d'%i,do_post_filtering=False)

def predict(batch_size=1):
    itr = vctk_loader.data_loader('test',batch_size,conditioning='unaligned_phonemes')
    count=0
    batch = np.random.choice(500)
    for j in xrange(5):
        for test_spk,test_X,test_X_mask,test_ctx,test_ctx_mask in itr:
            if test_X.shape[1]>500:
                break

    for i in tqdm(xrange(batch_size)):
        try:
            end_ctx = np.where(test_ctx[i]==0)[0][0]
        except IndexError:
            end_ctx = -1
        try:
            end_X = np.where(test_X_mask[i]==0)[0][0]
        except IndexError:
            end_X = -1
        pred_X = predict_fn(
              test_spk[i].reshape((1,)),
              test_ctx[i][None,:end_ctx]
            )
        save(pred_X[0],test_X[i,:end_X],i)

def char_to_audio(batch_size=1):
    phon_to_idx = pickle.load(open('/data/lisa/exp/kumarrit/vctk/phon2code.pkl'))
    idx_to_phon = {x:y for y,x in phon_to_idx.iteritems()}
    itr = vctk_loader.data_loader('test',batch_size,append_tokens=True,conditioning='text')
    count=0
    batch = np.random.choice(500)
    for j in xrange(2):
        for test_spk,test_X,test_X_mask,test_ctx,test_ctx_mask in itr:
            if test_X.shape[1]>400:
                break
    # for i in xrange(random.choice(range(20))):
    #     test_ctx,test_ctx_mask,_,_ = itr.next()

    for i in tqdm(xrange(batch_size)):
        try:
            end_ctx = np.where(test_ctx[i]==0)[0][0]
        except IndexError:
            end_ctx = -1
        try:
            end_X = np.where(test_X_mask[i]==0)[0][0]
        except IndexError:
            end_X = -1
        phons = nmt_fn(test_ctx[i,:end_ctx][None,:],test_ctx_mask[i,:end_ctx][None,:]).flatten()
        try:
            end_idx = np.where(phons==0)[0][0]
            phons = phons[:end_idx].tolist()
        except:
            phons = phons.tolist()
        print ' '.join([idx_to_phon[x] for x in phons])
        pred_X = predict_fn(
              test_spk[i].reshape((1,)),
              np.asarray(phons,dtype=np.int32).reshape((1,-1))
            )
        # save(pred_X[0],test_X[i,:end_X],i)
        generate.generate_wav(pred_X[0],base='generated_sample_%d'%i,do_post_filtering=False)


def new_sentence(text,PATH='/data/lisa/exp/kumarrit/vctk'):
    import nltk
    import re
    import random
    model = nltk.corpus.cmudict.dict()
    tknzr = nltk.tokenize.WordPunctTokenizer()
    tokens = tknzr.tokenize(text.lower())
    char_to_idx = pickle.load(open(PATH+'/phon2code.pkl'))
    result = []
    for tkn in tokens:
        if tkn in ['.',',']:
            result += ['ssil']
        else:
            result += [re.sub('\d+','',c.lower()) for c in random.choice(model[tkn])]
    print result
    ctx = np.asarray([char_to_idx[x] for x in result]).reshape((1,-1)).astype('int32')
    pred_X = predict_fn(
          np.asarray([np.random.choice(N_SPEAKERS),]).reshape((1,)).astype('int32'),
          ctx
        )
    generate.generate_wav(pred_X[0],base='generated_sample_16',do_post_filtering=False)
# new_sentence("GET A GLIMPSE OF WHAT THE LYRE BIRD A P I WILL OFFER")
print "Compiled Function!"

lib.load_params('vctk_synthesizer_GRU_l1.pkl')
lib.load_params('vctk_duration_predictor_best.pkl')
lib.load_params('vctk_nmt_best.pkl')

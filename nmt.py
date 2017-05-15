import theano
import theano.tensor as T
import lib
import lib.ops
import numpy as np
import time
import lasagne
import vctk_loader

BATCH_SIZE = 16
N_CHARS = 37
N_PHONS = 45
NB_EPOCHS = 50
LR = 0.001
N_LAYERS = 4
GRAD_CLIP = 1
SAVE_FILE_NAME = 'vctk_nmt_best.pkl'

chars = T.imatrix()
phons = T.imatrix()
chars_mask = T.fmatrix()
phons_mask = T.fmatrix()

learn_rate = T.fscalar()
drop_prob = T.fscalar()

def train_nmt(chars,chars_mask,phons):
    emb_chars = lib.ops.Embedding(
        'NMT.Embedding_Chars',
        N_CHARS,
        256,
        chars
    )
    emb_phons = lib.ops.Embedding(
        'NMT.Embedding_Phons',
        N_PHONS,
        256,
        phons
    )
    enc = lib.ops.BiRNN(
        'GRU',
        'NMT.Encoder',
        emb_chars,
        256,
        128,
        mask=chars_mask
    )[:,:,0]
    out1,out2 = lib.ops.AttnDec(
        'NMT.AttentionalDecoder',
        enc,
        256,
        N_PHONS,
        256,
        256,
        inputs=emb_phons,
        mode='train'
    )
    readout = lib.ops.Linear(
        'NMT.AttentionalDecoder.Output.MLP.1',
        T.concatenate([emb_phons,out1[:,:,:256],out2],-1),
        256+256+256,
        N_PHONS
    )
    return readout

def test_nmt(chars,chars_mask):
    emb_chars = lib.ops.Embedding(
        'NMT.Embedding_Chars',
        N_CHARS,
        256,
        chars
    )
    enc = lib.ops.BiRNN(
        'GRU',
        'NMT.Encoder',
        emb_chars,
        256,
        128,
        mask=chars_mask
    )[:,:,0]
    readout = lib.ops.AttnDec(
        'NMT.AttentionalDecoder',
        enc,
        256,
        N_PHONS,
        256,
        256,
        inputs=None,
        mode='open-loop'
    )
    return readout


def score(batch_size=16):
    start = time.time()
    valid_costs = []
    valid_itr = vctk_loader.nmt_data_loader('valid',batch_size,dataset='vctk')
    costs = []
    times = []
    for chars,chars_mask,phons,phons_mask in valid_itr:
        valid_loss = test_fn(
                    chars,
                    chars_mask,
                    phons,
                    phons_mask
                )
        valid_costs.append(valid_loss)
    print "Validation Completed! cost: {} time: {}".format(np.mean(np.asarray(valid_costs),axis=0),time.time()-start)
    return np.mean(valid_costs)

def test(batch_size=1):
    import random
    phon_to_idx = pickle.load(open('/data/lisa/exp/kumarrit/vctk/phon2code.pkl'))
    char_to_idx = pickle.load(open('/data/lisa/exp/kumarrit/vctk/char2code.pkl'))
    idx_to_phon = {x:y for y,x in phon_to_idx.iteritems()}
    idx_to_char = {x:y for y,x in char_to_idx.iteritems()}
    idx_to_char[36] = '#START'
    idx_to_char[0] = '#END'
    test_itr = vctk_loader.nmt_data_loader('test',batch_size)
    for i in xrange(random.choice(range(20))):
        chars,chars_mask,phons,phons_mask = test_itr.next()
    preds = predict_fn(chars,chars_mask).flatten()
    end_idx = np.where(preds==0)[0][0]
    preds = preds[:end_idx].tolist()
    print [idx_to_phon[x] for x in preds]
    print ''.join([idx_to_char[x] for x in chars.flatten().tolist()])


if __name__=='__main__':
    batch_size = T.shape(chars)[0]
    logits = train_nmt(chars,chars_mask,phons[:,:-1])
    preds = test_nmt(chars,chars_mask)
    loss = T.nnet.categorical_crossentropy(
        T.nnet.softmax(logits.reshape((-1,N_PHONS))),
        phons[:,1:].flatten()
    ).reshape((batch_size,-1))
    cost = T.sum(loss*phons_mask[:,1:])/T.sum(phons_mask[:,1:])

    params = lib.search(cost, lambda x: hasattr(x, "param"))
    lib.print_params_info(params)
    grads = T.grad(cost, wrt=params, disconnected_inputs='warn')
    grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in grads]

    print "Gradients Computed"
    updates = lasagne.updates.adam(grads, params, learning_rate=learn_rate)

    train_fn = theano.function(
        [chars,chars_mask,phons,phons_mask,learn_rate],
        cost,
        updates=updates,
        on_unused_input='warn'
    )

    test_fn = theano.function(
        [chars,chars_mask,phons,phons_mask],
        cost,
        on_unused_input='warn'
    )

    predict_fn = theano.function(
        [chars,chars_mask],
        preds
    )

    print "Compiled Functions!"
    i=0
    best_cost = 1000.
    for i in xrange(i, NB_EPOCHS):
        iteration = 0
        costs = []
        times = []

        itr = vctk_loader.nmt_data_loader('train',BATCH_SIZE,dataset='vctk')
        for a,b,c,d in itr:
            start = time.time()
            iteration += 1
            loss = train_fn(
                    a,
                    b,
                    c,
                    d,
                    LR
                )

            times.append(time.time()-start)
            costs.append(loss)
            if iteration%50==0:
                print "Iteration: {} (Epoch {})! cost: {} time: {}" .format(iteration, i + 1, np.mean(np.asarray(costs),axis=0), np.mean(times))

        print "Epoch {} Completed! cost: {} time: {}" .format(i + 1,np.mean(np.asarray(costs),axis=0),np.mean(times))
        valid_cost = score()
        if valid_cost < best_cost:
            best_cost = valid_cost
            lib.save_params(SAVE_FILE_NAME)
            print "Saving Model!"

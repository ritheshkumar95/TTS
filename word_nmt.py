import theano
import theano.tensor as T
import lib
import lib.ops
import numpy as np
import time
import lasagne
import nltk
import re
import random
import pickle

def data_loader(set='train',batch_size=16):
    phon_to_idx = pickle.load(open('/data/lisa/exp/kumarrit/vctk/phon2code.pkl'))
    char_to_idx = pickle.load(open('/data/lisa/exp/kumarrit/vctk/char2code.pkl'))
    # phon_to_idx = pickle.load(open('phon2code.pkl'))
    # char_to_idx = pickle.load(open('char2code.pkl'))
    model = nltk.corpus.cmudict.dict()
    data = np.asarray(list(model.iteritems()),dtype=object)

    remove_num = lambda c: re.sub('\d+','',c.lower())
    def pad_and_mask(data,end_idx):
        max_len = max([len(x) for x in data])+2
        new_data = np.zeros((len(data),max_len)).astype('int32')
        mask = np.zeros((len(data),max_len)).astype('float32')
        for i in xrange(len(data)):
            new_data[i,1:len(data[i])+1] = data[i]
            # new_data[i,1+len(data[i])] = end_idx
            new_data[i,0] = end_idx
            mask[i,:len(data[i])+2] = 1.
        return new_data,mask

    total_len = len(data)
    np.random.seed(111)
    idxs = range(total_len)
    np.random.shuffle(idxs)
    ranges = {}
    ranges['train'] = (0,int(.8*len(idxs)))
    ranges['valid'] = (int(0.8*len(idxs)),int(0.9*len(idxs)))
    ranges['test'] = (int(0.9*len(idxs)),len(idxs))
    idxs = idxs[slice(*ranges[set])]
    N_FILES = (len(idxs)//batch_size)*batch_size

    skip_count=0
    for i in xrange(0,N_FILES,batch_size):
        batch = data[idxs[i:i+batch_size]]
        try:
            x_t = [[char_to_idx[y] for y in list(x[0])] for x in batch]
        except KeyError as e:
            skip_count += 1
            # print "Skipping ",e, "({})".format(skip_count)
        y_t = [[phon_to_idx[remove_num(y)] for y in random.choice(x[1])] for x in batch]
        chars,chars_mask = pad_and_mask(x_t,36)
        phons,phons_mask = pad_and_mask(y_t,44)
        yield chars,chars_mask,phons,phons_mask

theano.tensor.cmp_sloppy=2

BATCH_SIZE = 32
N_CHARS = 37
N_PHONS = 45
NB_EPOCHS = 50
LR = 0.001
N_LAYERS = 4
GRAD_CLIP = 1
SAVE_FILE_NAME = 'cmudict_nmt_best.pkl'

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
    valid_itr = data_loader('valid',batch_size)
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

        itr = data_loader('train',BATCH_SIZE)
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
            if iteration%500==0:
                print "Iteration: {} (Epoch {})! cost: {} time: {}" .format(iteration, i + 1, np.mean(np.asarray(costs),axis=0), np.mean(times))

        print "Epoch {} Completed! cost: {} time: {}" .format(i + 1,np.mean(np.asarray(costs),axis=0),np.mean(times))
        valid_cost = score()
        if valid_cost < best_cost:
            best_cost = valid_cost
            lib.save_params(SAVE_FILE_NAME)
            print "Saving Model!"

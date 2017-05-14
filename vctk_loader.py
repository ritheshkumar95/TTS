import numpy as np
import h5py
from itertools import groupby

def get_data_idxs(PATH,set):
    data = h5py.File(PATH)
    total_len = data['phonemes'].shape[0]
    np.random.seed(111)
    idxs = range(total_len)
    np.random.shuffle(idxs)
    ranges = {}
    ranges['train'] = (0,int(.8*len(idxs)))
    ranges['valid'] = (int(0.8*len(idxs)),int(0.9*len(idxs)))
    ranges['test'] = (int(0.9*len(idxs)),len(idxs))
    idxs = idxs[slice(*ranges[set])]
    print "{} indexes starts at {} with total of {} data points!".format(set,idxs[:5],len(idxs))
    return data,idxs

def data_loader(set='train',batch_size=32,use_speaker=True,append_tokens=False,conditioning='phonemes',dataset='vctk'):
    '''
        Data Generator for VCTK Dataset
        :returns:
            reset              : vector of size (batch_size,) indicating if initiate states must be reset
                   (used along with truncate bptt)
            speaker index      : vector of size (batch_size,) indicating index of speaker
            vocoder features   : 3-D tensor of shape (batch_size, max_n_feats, 63). It is padded along 2nd dimension to train along mini-batch
            mask               : integer matrix of shape (batch_size, max_n_feats). It provides a mask for vocoder features that indicate real vs padded data
            conditioning vector: integer matrix of shape (batch_size, max_cond_len) where each element represents index of phoneme / character.
                                 conditioning vector is padded with zeros to match maximum length along mini-batch

                                 NOTE: Explicit mask is not required for (padded) conditioning vector as attention computations become zeros by default along padded data

        N_CHARS = 1-35 [Use 0 for <start> and 36 for end] [Total: 36 or 37]
        N_PHONEMES = 1-43 [Use 0 for <start> and 44 for end] [Total 44 or 45]

    '''
    PATH = '/data/lisa/exp/kumarrit/{}/{}.hdf5'.format(dataset,dataset)
    data,idxs = get_data_idxs(PATH,set)
    N_FILES = (len(idxs)//batch_size)*batch_size
    for i in xrange(0,N_FILES,batch_size):
        sorted_idxs = sorted(idxs[i: i+batch_size])
        features = data['features'][sorted_idxs]
        features_shapes = data['features_shapes'][sorted_idxs]
        cond_vector = data[conditioning][sorted_idxs]
        max_feat_shape = max([x[0] for x in features_shapes])

        X = np.zeros((batch_size,max_feat_shape,63)).astype('float32')
        X_mask = np.zeros((batch_size,max_feat_shape)).astype('float32')
        for j in xrange(batch_size):
            features[j] = features[j].reshape((-1,63))
            cur_len = features[j].shape[0]
            X[j,:cur_len,:] = features[j]
            X_mask[j,:cur_len] = 1.

        max_ctx_shape = max([x.shape[0] for x in cond_vector])
        assert max_feat_shape==max_ctx_shape,"Mismatch error"
        if append_tokens:
            max_ctx_shape += 2
            start_token = 36 if conditioning=='text' else 44
            ctx = np.zeros((batch_size,max_ctx_shape)).astype('int32')
            ctx_mask = np.zeros((batch_size,max_ctx_shape)).astype('float32')
            for j in xrange(batch_size):
                cur_len = cond_vector[j].shape[0]
                ctx[j,1:1+cur_len] = cond_vector[j]
                ctx[j,0] = start_token
                ctx_mask[j,:cur_len+2] = 1.
        else:
            ctx = np.zeros((batch_size,max_ctx_shape)).astype('int32')
            ctx_mask = np.zeros((batch_size,max_ctx_shape)).astype('float32')
            for j in xrange(batch_size):
                cur_len = cond_vector[j].shape[0]
                ctx[j,:cur_len] = cond_vector[j]
                ctx_mask[j,:cur_len] = 1.

        spk_ids = data['speaker_index'][sorted_idxs][:,0] #subtract 1 for 0 indexing

        if use_speaker:
            yield (
                   spk_ids,
                   X,
                   X_mask,
                   ctx,
                   ctx_mask
                   )
        else:
            yield (
                   X,
                   X_mask,
                   ctx,
                   ctx_mask
                  )


def duration_data_loader(set='train',batch_size=32,dataset='vctk'):
    '''
        Data Generator for VCTK Dataset
        N_CHARS = 1-35 [Use 0 for <start> and 36 for end] [Total: 36 or 37]
        N_PHONEMES = 1-43 [Use 0 for <start> and 44 for end] [Total 44 or 45]
    '''
    PATH = '/data/lisa/exp/kumarrit/{}/{}.hdf5'.format(dataset,dataset)
    data,idxs = get_data_idxs(PATH,set)
    N_FILES = (len(idxs)//batch_size)*batch_size
    for i in xrange(0,N_FILES,batch_size):
        sorted_idxs = sorted(idxs[i: i+batch_size])
        train_labs = data['phonemes'][sorted_idxs]
        train_spk = data['speaker_index'][sorted_idxs]
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

        yield np.asarray(labels,dtype=np.int32),np.asarray(durations,dtype=np.int32),mask

def nmt_data_loader(set='train',batch_size=32,dataset='vctk'):
    '''
        N_CHARS = 1-35 [Use 0 for <start> and 36 for end] [Total: 36 or 37]
        N_PHONEMES = 1-43 [Use 0 for <start> and 44 for end] [Total 44 or 45]
    '''
    PATH = '/data/lisa/exp/kumarrit/{}/{}.hdf5'.format(dataset,dataset)
    data,idxs = get_data_idxs(PATH,set)
    N_FILES = (len(idxs)//batch_size)*batch_size

    def pad_and_mask(data,end_idx):
        max_len = max([len(x) for x in data])+2
        new_data = np.zeros((data.shape[0],max_len)).astype('int32')
        mask = np.zeros((data.shape[0],max_len)).astype('float32')
        for i in xrange(data.shape[0]):
            new_data[i,1:len(data[i])+1] = data[i]
            # new_data[i,1+len(data[i])] = end_idx
            new_data[i,0] = end_idx
            mask[i,:len(data[i])+2] = 1.
        return new_data,mask


    for i in xrange(0,N_FILES,batch_size):
        sorted_idxs = sorted(idxs[i: i+batch_size])
        phons = data['unaligned_phonemes'][sorted_idxs]
        chars = data['text'][sorted_idxs]

        phons,phons_mask = pad_and_mask(phons,44)
        chars,chars_mask = pad_and_mask(chars,36)

        yield chars,chars_mask,phons,phons_mask

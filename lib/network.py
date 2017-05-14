import lib
import lib.ops
import theano.tensor as T
import theano
import numpy as np

def alex_net(inp,DIM=512):
    X = T.nnet.relu(lib.ops.conv2d('conv1', inp, 11, 4, 1, 96, pad = 'half'))
    X = lib.ops.max_pool('pool1', X, k=3, s=2)
    #X = lib.ops.Batchnorm('BatchNorm1', 96, X, axes=[0,2,3])

    X = T.nnet.relu(lib.ops.conv2d('conv2', X, 5, 1, 96, 256, pad = 'half'))
    X = lib.ops.max_pool('pool2', X, k=3, s=2)
    #X = lib.ops.Batchnorm('BatchNorm2', 256, X, axes=[0,2,3])

    X = T.nnet.relu(lib.ops.conv2d('conv3', X, 3, 1, 256, 384, pad = 'half'))
    X = T.nnet.relu(lib.ops.conv2d('conv4', X, 3, 1, 384, 384, pad = 'half'))
    X = T.nnet.relu(lib.ops.conv2d('conv5', X, 3, 1, 384, 256, pad = 'half'))
    X = lib.ops.max_pool('pool5', X, k=3, s=2)
    #X = lib.ops.BatchNorm('BatchNorm5', X, lsize=5)

    X = T.nnet.relu(lib.ops.Linear('fc6', X.reshape((X.shape[0],-1)),8192,4096))
    X = lib.ops.dropout(X,0.5)

    X = T.nnet.relu(lib.ops.Linear('fc7',X,4096,4096))
    X = lib.ops.dropout(X,0.5)

    X = lib.ops.Linear('fc8',X,4096,DIM)

    return X


def vgg16(X,num_feats=64):
    X = T.nnet.relu(lib.ops.conv2d('conv1_1', X, 3, 1, 1, num_feats, pad = 'half'))
    X = T.nnet.relu(lib.ops.conv2d('conv1_2', X, 3, 1, num_feats, num_feats, pad = 'half'))
    X = lib.ops.max_pool('pool1', X, k=2, s=2)

    X = T.nnet.relu(lib.ops.conv2d('conv2_1', X, 3, 1, num_feats, 2*num_feats, pad = 'half'))
    X = T.nnet.relu(lib.ops.conv2d('conv2_2', X, 3, 1, 2*num_feats, 2*num_feats, pad = 'half'))
    X = lib.ops.max_pool('pool2', X, k=2, s=2)

    X = T.nnet.relu(lib.ops.conv2d('conv3_1', X, 3, 1, 2*num_feats, 4*num_feats, pad = 'half'))
    X = T.nnet.relu(lib.ops.conv2d('conv3_2', X, 3, 1, 4*num_feats, 4*num_feats, pad = 'half'))
    X = T.nnet.relu(lib.ops.conv2d('conv3_3', X, 3, 1, 4*num_feats, 4*num_feats, pad = 'half'))
    X = lib.ops.max_pool('pool3', X, k=2, s=2)

    X = T.nnet.relu(lib.ops.conv2d('conv4_1', X, 3, 1, 4*num_feats, 8*num_feats, pad = 'half'))
    X = T.nnet.relu(lib.ops.conv2d('conv4_2', X, 3, 1, 8*num_feats, 8*num_feats, pad = 'half'))
    X = T.nnet.relu(lib.ops.conv2d('conv4_3', X, 3, 1, 8*num_feats, 8*num_feats, pad = 'half'))
    X = lib.ops.max_pool('pool4', X, k=2, s=2)

    X = T.nnet.relu(lib.ops.conv2d('conv5_1', X, 3, 1, 8*num_feats, 8*num_feats, pad = 'half'))
    X = T.nnet.relu(lib.ops.conv2d('conv5_2', X, 3, 1, 8*num_feats, 8*num_feats, pad = 'half'))
    X = T.nnet.relu(lib.ops.conv2d('conv5_3', X, 3, 1, 8*num_feats, 8*num_feats, pad = 'half'))

    return X

import theano.tensor as T
import theano
import lib
import lib.ops
import numpy as np
import time
from tqdm import tqdm
import lasagne


def rmsprop(loss_or_grads, params, learning_rate=1.0, rho=0.9, epsilon=1e-6, sign='-', clip=False, clip_val=1.):
    grads = loss_or_grads
    from collections import OrderedDict

    updates = OrderedDict()

    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = rho * accu + (one - rho) * grad ** 2
        updates[accu] = accu_new
        if sign=='-':
            updated_param = param - (learning_rate * grad / T.sqrt(accu_new + epsilon))
        else:
            updated_param = param + (learning_rate * grad / T.sqrt(accu_new + epsilon))

        if clip:
            updated_param = T.clip(updated_param,lib.floatX(-clip_val),lib.floatX(clip_val))
        updates[param] = updated_param
    return updates

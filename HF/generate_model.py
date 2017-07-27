'''
Created on 2016/10/03

@author: Satoshi Kuroki
(Referened scripts of Nicolas Boulanger-Lewandowski, University of Montreal, 2012-2013)

Script for generate RNN model (Mante et al., 2013, Nature)
'''

import numpy as np
import theano
import theano.tensor as T

# model of original setting to (Mante et al., 2013, Nature)
def model_original(nx=4, nh=100, ny=1, p=None, tau = 10., seed = 0):
    np.random.seed(seed)
    if p == None:
        Wx = theano.shared(np.random.normal(0., 0.5, (nx, nh)).astype(theano.config.floatX))
        Wh = theano.shared(np.random.normal(0., 1./nh, (nh, nh)).astype(theano.config.floatX))
        Wy = theano.shared(np.zeros((nh,ny), dtype=theano.config.floatX))
        bh = theano.shared(np.zeros(nh, dtype=theano.config.floatX))
        by = theano.shared(np.zeros(ny, dtype=theano.config.floatX))
        p = [Wx, Wh, Wy, bh, by]
    else:
        Wx = p[0]; Wh = p[1]; Wy = p[2]; bh = p[3]; by = p[4]

    h0 = theano.shared(np.zeros(nh, dtype=theano.config.floatX))
    x = T.matrix('input_x')
    rho_h = T.matrix('rho_h')
    t = T.scalar('teachSig')
    #theano.config.exception_verbosity='high'
    def recurrence(x_t, rho_h_t, h_tm1):
        dh = (-h_tm1 + T.dot(x_t, Wx) + T.dot(h_tm1, Wh) + bh + rho_h_t) / tau
        ha_t = h_tm1 + dh
        h_t = T.tanh(ha_t)
        s_t = T.dot(h_t, Wy) + by
        return [ha_t, h_t, s_t]

    ([ha, h, y], updates) = theano.scan(fn=recurrence, sequences=[x, rho_h], outputs_info=[dict(), h0, dict()])

    h = T.tanh(ha)
    y_0 = y[0, 0]
    y_T = y[-1, 0]
    loss = (((0.-y_0) ** 2.) + ((t-y_T) ** 2.)) / 2.
    acc = T.neq(T.sgn(y_T), t)
    return p, [x, rho_h, t], y_T, [loss, acc], h, ha, y


# model to modify acitvities of neuronal units by variable 'mod'
def model_mod_act(nx=4, nh=100, ny=1, p=None, tau = 10., seed = 0):
    np.random.seed(seed)
    if p == None:
        Wx = theano.shared(np.random.normal(0., 0.5, (nx, nh)).astype(theano.config.floatX))
        Wh = theano.shared(np.random.normal(0., 1./nh, (nh, nh)).astype(theano.config.floatX))
        Wy = theano.shared(np.zeros((nh,ny), dtype=theano.config.floatX))
        bh = theano.shared(np.zeros(nh, dtype=theano.config.floatX))
        by = theano.shared(np.zeros(ny, dtype=theano.config.floatX))
        p = [Wx, Wh, Wy, bh, by]
    else:
        Wx = p[0]; Wh = p[1]; Wy = p[2]; bh = p[3]; by = p[4]

    h0 = theano.shared(np.zeros(nh, dtype=theano.config.floatX))
    x = T.matrix('input_x')
    rho_h = T.matrix('rho_h')
    t = T.scalar('teachSig')
    mod = T.matrix('modulator')
    #theano.config.exception_verbosity='high'
    def recurrence(x_t, rho_h_t, mod_t, h_tm1):
        dh = (-h_tm1 + bh  + mod_t * (T.dot(x_t, Wx) + T.dot(h_tm1, Wh) + rho_h_t)) / tau
        ha_t = h_tm1 + dh
        h_t = T.tanh(ha_t)
        s_t = T.dot(h_t, Wy) + by
        return [ha_t, h_t, s_t]

    ([ha, h, y], updates) = theano.scan(fn=recurrence, sequences=[x, rho_h, mod], outputs_info=[dict(), h0, dict()])

    h = T.tanh(ha)
    y_0 = y[0, 0]
    y_T = y[-1, 0]
    loss = (((0.-y_0) ** 2.) + ((t-y_T) ** 2.)) / 2.
    acc = T.neq(T.sgn(y_T), t)
    return p, [x, rho_h, mod, t], y_T, [loss, acc], h, ha, y

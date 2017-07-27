# -*- coding: utf-8 -*-
"""
Created on Thu Mar 02 14:25:25 2017

@author: Satoshi Kuroki

Script for inactivation experiment
"""

from __future__ import absolute_import
from __future__ import division

import cPickle as pickle
import os
import shutil
import sys
import time
from   collections import OrderedDict

import numpy as np

from .utils import print_settings
from .rnn   import RNN

#=========================================================================================
# Activation functions
#=========================================================================================

def rectify(x):
    return x*(x > 0)

def rectify_power(x, n=2):
    return x**n*(x > 0)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def rtanh(x):
    return np.tanh(rectify(x))

def softmax(x):
    """
    Softmax function.

    x : 2D numpy.ndarray
        The outputs must be in the second dimension.

    """
    e = np.exp(x)
    return e/np.sum(e, axis=1, keepdims=True)

activation_functions = {
    'linear':        lambda x: x,
    'rectify':       rectify,
    'rectify_power': rectify_power,
    'sigmoid':       sigmoid,
    'tanh':          np.tanh,
    'rtanh':         rtanh,
    'softmax':       softmax
}

#=========================================================================================

def euler_inact(alpha, x_t, r_t, Win, Wrec, brec, bout, u, noise_rec,
                f_hidden, r, w_inact):
    for i in xrange(1, r.shape[0]):
        x_t += alpha*(-x_t            # Leak
                      + Wrec.dot(r_t) # Recurrent input
                      + brec          # Bias
                      + Win.dot(u[i]) # Input
                      + noise_rec[i]) # Recurrent noise
        x_t *= w_inact # added by sk 2017/3/2
        r_t = f_hidden(x_t)

        r[i] = r_t

class RNN_inact(RNN):
    def __init__(self, savefile=None, rnnparams={}, verbose=True):
        RNN.__init__(self, savefile, rnnparams, verbose)

    def run(self, id_inact=None, T=None, inputs=None, rng=None, seed=1234):
        """
        Run the network.

        Parameters
        ----------

        T : float, optional
            Duration for which to run the network. If `None`, `inputs` must not be
            `None` so that the network can be run for the trial duration.

        inputs : (generate_trial, params), optional

        rng : numpy.random.RandomState
              Random number generator. If `None`, one will be created using seed.

        seed : int, optional
               Seed for the random number generator.

        """
        if self.verbose:
            config = OrderedDict()

            config['dt']        = '{} ms'.format(self.p['dt'])
            config['threshold'] = self.p['threshold']

            print_settings(config)

        # Random number generator
        if rng is None:
            rng = np.random.RandomState(seed)

        #---------------------------------------------------------------------------------
        # Setup
        #---------------------------------------------------------------------------------

        N           = self.p['N']
        Nin         = self.p['Nin']
        Nout        = self.p['Nout']
        baseline_in = self.p['baseline_in']
        var_in      = self.p['var_in']
        var_rec     = self.p['var_rec']
        dt          = self.p['dt']
        tau         = self.p['tau']
        tau_in      = self.p.get('tau_in', self.p['tau'])
        sigma0      = self.p['sigma0']
        mode        = self.p['mode']

        # Check dt
        if np.any(dt > tau/10):
            print("[ {}.RNN.run ] Warning: dt seems a bit large.".format(THIS))

        # Float
        dtype = self.Wrec[0,0]

        #---------------------------------------------------------------------------------
        # External input
        #---------------------------------------------------------------------------------

        if inputs is None:
            if T is None:
                raise RuntimeError("[ {}.RNN.run ] Cannot determine the trial duration."
                                   .format(THIS))

            self.t = np.linspace(0, T, int(T/dt)+1).astype(dtype)
            if self.Win is not None:
                u = np.zeros((len(self.t), Nin), dtype=dtype)
            info = None
        else:
            generate_trial, params = inputs

            trial  = generate_trial(rng, dt, params)
            info   = trial['info']
            self.t = np.concatenate(([0], trial['t']))

            u = np.zeros((len(self.t), trial['inputs'].shape[1]), dtype=dtype)
            u[1:,:] = trial['inputs']

            info['epochs'] = trial['epochs']

        Nt = len(self.t)

        # added by sk 2017/3/2
        self.w_inact = np.ones(self.x0.shape, dtype=dtype)
        if id_inact is not None:
            self.w_inact[id_inact] = 0.

        #---------------------------------------------------------------------------------
        # Variables to record
        #---------------------------------------------------------------------------------

        if self.Win is not None:
            self.u = np.zeros((Nt, Nin), dtype=dtype)
        else:
            self.u = None
        self.r = np.zeros((Nt, N), dtype=dtype)
        self.z = np.zeros((Nt, Nout), dtype=dtype)

        #---------------------------------------------------------------------------------
        # Activation functions
        #---------------------------------------------------------------------------------

        f_hidden = activation_functions[self.p['hidden_activation']]
        f_output = activation_functions[self.p['output_activation']]

        #---------------------------------------------------------------------------------
        # Integrate
        #---------------------------------------------------------------------------------

        # Time step
        alpha = dt/tau

        # Input noise
        if self.Win is not None:
            var_in = 2*tau_in/dt*var_in
            if np.isscalar(var_in) or var_in.ndim == 1:
                if np.any(var_in > 0):
                    noise_in = np.sqrt(var_in)*rng.normal(size=(Nt, Nin))
                else:
                    noise_in = np.zeros((Nt, Nin))
            else:
                noise_in = rng.multivariate_normal(np.zeros(Nin), var_in, Nt)
            noise_in = np.asarray(noise_in, dtype=dtype)

        # Recurrent noise
        var_rec = 2/dt*var_rec
        if np.isscalar(var_rec) or var_rec.ndim == 1:
            if np.any(var_rec > 0):
                noise_rec = np.sqrt(var_rec)*rng.normal(size=(Nt, N))
            else:
                noise_rec = np.zeros((Nt, N))
        else:
            noise_rec = rng.multivariate_normal(np.zeros(N), var_rec, Nt)
        noise_rec = np.asarray(np.sqrt(tau)*noise_rec, dtype=dtype)

        # Inputs
        if self.Win is not None:
            self.u = baseline_in + u + noise_in
            if self.p['rectify_inputs']:
                self.u = rectify(self.u)

        # Initial conditions
        if hasattr(self, 'x_last'):
            if self.verbose:
                print("[ {}.RNN.run ] Continuing from previous run.".format(THIS))
            x_t = self.x_last.copy()
        else:
            x_t = self.x0.copy()
            if sigma0 > 0:
                x_t += sigma0*rng.normal(size=N)
        r_t = f_hidden(x_t)

        # Record initial conditions
        self.r[0] = r_t

        # Integrate
        if np.isscalar(alpha):
            alpha = alpha*np.ones(N, dtype=dtype)
        if self.Win is not None:
            # modifyed by sk 2017/3/2
            euler_inact(alpha, x_t, r_t, self.Win, self.Wrec, self.brec,
                   self.bout, self.u, noise_rec, f_hidden, self.r, self.w_inact)

        if self.Wout is not None:
            self.z = f_output(self.r.dot(self.Wout.T) + self.bout)
        else:
            self.z = self.r

        # Transpose so first dimension is units
        if self.u is not None:
            self.u = self.u.T
        self.r = self.r.T
        self.z = self.z.T

        # In continuous mode start from here the next time
        if mode == 'continuous':
            self.x_last = x_t

        #---------------------------------------------------------------------------------

        return info

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 17:00:43 2017

@author: Satoshi Kuroki

Script for inactivation experiment
"""

from __future__ import absolute_import, division

import os
from   collections import OrderedDict
import datetime
import sys

import numpy as np

import theano
from   theano import tensor

from .         import nptools, tasktools, theanotools, utils
from .debug    import DEBUG
from .networks import Networks
from .sgd      import Adam
from .policygradient import PolicyGradient
from .model import Model

def behaviorfile(path,tag):
    return os.path.join(path, 'trials_behavior_inact'+tag+'.pkl')

def activityfile(path,tag):
    return os.path.join(path, 'trials_activity_inact'+tag+'.pkl')

def run(action, trials, pgi, scratchpath,id_inact=None, dt_save=None):
    if dt_save is not None:
        dt  = pgi.dt
        inc = int(dt_save/dt)
    else:
        inc = 1
    print("Saving in increments of {}".format(inc))

    # added by sk 2017/2/28
    if id_inact is not None:
        n_ids = len(id_inact)
        str_ids = 'inact_n' + str(n_ids)
        for i in xrange(n_ids):
            if i < 5:
                str_ids = str_ids + '_' + str(id_inact[i])
    else:
        n_ids = 0
        str_ids = 'inact_n' + str(n_ids) + '_full'
    print(str_ids)

    # Run trials
    print("Saving behavior only.")
    trialsfile = behaviorfile(scratchpath,str_ids)

    (U, Q, Q_b, Z, Z_b, A, R, M, init, init_b, states_0, states_0_b,
     perf) = pgi.run_trials(trials, id_inact=id_inact, progress_bar=True)

    for trial in trials:
        trial['time'] = trial['time'][::inc]
    save = [trials, A[::inc], R[::inc], M[::inc], perf]

    # Performance
    perf.display()

    # Save
    utils.save(trialsfile, save)

    # File size
    size_in_bytes = os.path.getsize(trialsfile)
    print("File size: {:.1f} MB".format(size_in_bytes/2**20))


class PolicyGradient_inact(PolicyGradient):
    def __init__(self, Task, config_or_savefile, seed, dt=None, load='best'):
        PolicyGradient.__init__(self, Task, config_or_savefile, seed, dt, load)
    def run_trials(self, trials, id_inact = None, id_b_inact=None, init=None, init_b=None,
                   return_states=False, perf=None, task=None, progress_bar=False,
                   p_dropout=0):

        if isinstance(trials, list):
            n_trials = len(trials)
        else:
            n_trials = trials
            trials   = []

        if return_states:
            run_value_network = True
        else:
            run_value_network = False

        # Storage
        U   = theanotools.zeros((self.Tmax, n_trials, self.Nin))
        Z   = theanotools.zeros((self.Tmax, n_trials, self.Nout))
        A   = theanotools.zeros((self.Tmax, n_trials, self.n_actions))
        R   = theanotools.zeros((self.Tmax, n_trials))
        M   = theanotools.zeros((self.Tmax, n_trials))
        Z_b = theanotools.zeros((self.Tmax, n_trials))

        # Noise
        Q   = self.make_noise((self.Tmax, n_trials, self.policy_net.noise_dim),
                               self.scaled_var_rec)
        Q_b = self.make_noise((self.Tmax, n_trials, self.baseline_net.noise_dim),
                               self.scaled_baseline_var_rec)

        x_t   = theanotools.zeros((1, self.policy_net.N))
        x_t_b = theanotools.zeros((1, self.baseline_net.N))

        # added by sk 2017/2/28
        w_inact   = np.ones((1, self.policy_net.N), dtype=theano.config.floatX)
        w_b_inact = np.ones((1, self.baseline_net.N), dtype=theano.config.floatX)
        if not id_inact   == None:
            w_inact[0, id_inact]     = 0.
        if not id_b_inact == None:
            w_b_inact[0, id_b_inact] = 0.

        # Dropout mask
        #D   = np.ones((self.Tmax, n_trials, self.policy_net.N))
        #D_b = np.ones((self.Tmax, n_trials, self.baseline_net.N))
        #if p_dropout > 0:
        #    D   -= (np.uniform(size=D.shape) < p_dropout)
        #    D_b -= (np.uniform(size=D_b.shape) < p_dropout)

        # Firing rates
        if return_states:
            r_policy = theanotools.zeros((self.Tmax, n_trials, self.policy_net.N))
            r_value  = theanotools.zeros((self.Tmax, n_trials, self.baseline_net.N))

        # Keep track of initial conditions
        if self.mode == 'continuous':
            x0   = theanotools.zeros((n_trials, self.policy_net.N))
            x0_b = theanotools.zeros((n_trials, self.baseline_net.N))
        else:
            x0   = None
            x0_b = None

        # Performance
        if perf is None:
            perf = self.Performance()

        # Setup progress bar
        if progress_bar:
            progress_inc  = max(int(n_trials/50), 1)
            progress_half = 25*progress_inc
            if progress_half > n_trials:
                progress_half = -1
            utils.println("[ PolicyGradient.run_trials ] ")

        for n in xrange(n_trials):
            if progress_bar and n % progress_inc == 0:
                if n == 0:
                    utils.println("0")
                elif n == progress_half:
                    utils.println("50")
                else:
                    utils.println("|")

            # Initialize trial
            if hasattr(self.task, 'start_trial'):
                self.task.start_trial()

            # Generate trials
            if n < len(trials):
                trial = trials[n]
            else:
                trial = self.task.get_condition(self.rng, self.dt)
                trials.append(trial)

            #-----------------------------------------------------------------------------
            # Time t = 0
            #-----------------------------------------------------------------------------

            t = 0
            if init is None:
                z_t,   x_t[0]   = self.policy_step_0()
                z_t_b, x_t_b[0] = self.baseline_step_0()
            else:
                z_t,   x_t[0]   = init
                z_t_b, x_t_b[0] = init_b
            Z[t,n]   = z_t
            Z_b[t,n] = z_t_b

            # Save initial condition
            if x0 is not None:
                x0[n]   = x_t[0]
                x0_b[n] = x_t_b[0]

            # Save states
            if return_states:
                r_policy[t,n] = self.policy_net.firing_rate(x_t[0])
                r_value[t,n]  = self.baseline_net.firing_rate(x_t_b[0])

            # Select action
            a_t = theanotools.choice(self.rng, self.Nout, p=np.reshape(z_t, (self.Nout,)))
            A[t,n,a_t] = 1

            #a_t = self.rng.normal(np.reshape(z_t, (self.Nout,)), self.sigma)
            #A[t,n,0] = a_t

            # Trial step
            U[t,n], R[t,n], status = self.task.get_step(self.rng, self.dt,
                                                        trial, t+1, a_t)
            u_t    = U[t,n]
            M[t,n] = 1

            # Noise
            q_t   = Q[t,n]
            q_t_b = Q_b[t,n]

            #-----------------------------------------------------------------------------
            # Time t > 0
            #-----------------------------------------------------------------------------

            for t in xrange(1, self.Tmax):
                # Aborted episode
                if not status['continue']:
                    break

                # Policy
                z_t, x_t[0] = self.policy_step_t(u_t[None,:], q_t[None,:], x_t)
                Z[t,n] = z_t

                # added by sk 2017/2/28
                #print 'pre', x_t[0,id_inact]
                x_t[0] = x_t[0] * w_inact
                #print 'post', x_t[0,id_inact]

                # Baseline
                r_t = self.policy_net.firing_rate(x_t[0])
                u_t_b = np.concatenate((r_t, A[t-1,n]), axis=-1)
                z_t_b, x_t_b[0] = self.baseline_step_t(u_t_b[None,:],
                                                       q_t_b[None,:],
                                                       x_t_b)
                Z_b[t,n] = z_t_b

                # added by sk 2017/2/28
                x_t_b[0] = x_t_b[0] * w_b_inact

                # Firing rates
                if return_states:
                    r_policy[t,n] = self.policy_net.firing_rate(x_t[0])
                    r_value[t,n]  = self.baseline_net.firing_rate(x_t_b[0])

                    #W = self.policy_net.get_values()['Wout']
                    #b = self.policy_net.get_values()['bout']
                    #V = r_policy[t,n].dot(W) + b
                    #print(t)
                    #print(V)
                    #print(np.exp(V))

                # Select action
                a_t = theanotools.choice(self.rng, self.Nout,
                                         p=np.reshape(z_t, (self.Nout,)))
                A[t,n,a_t] = 1
                #a_t = self.rng.normal(np.reshape(z_t, (self.Nout,)), self.sigma)
                #A[t,n,0] = a_t

                # Trial step
                if self.abort_on_last_t and t == self.Tmax-1:
                    U[t,n] = 0
                    R[t,n] = self.R_TERMINAL
                    status = {'continue': False, 'reward': R[t,n]}
                else:
                    U[t,n], R[t,n], status = self.task.get_step(self.rng, self.dt,
                                                                trial, t+1, a_t)
                R[t,n] *= self.discount_factor(t)

                u_t    = U[t,n]
                M[t,n] = 1

                # Noise
                q_t   = Q[t,n]
                q_t_b = Q_b[t,n]

            #-----------------------------------------------------------------------------

            # Update performance
            perf.update(trial, status)

            # Save next state if necessary
            if self.mode == 'continuous':
                init   = self.policy_step_t(u_t[None,:], q_t[None,:], x_t)
                init_b = self.baseline_step_t(u_t_b[None,:], q_t_b[None,:], x_t_b)
        if progress_bar:
            print("100")

        #---------------------------------------------------------------------------------

        rvals = [U, Q, Q_b, Z, Z_b, A, R, M, init, init_b, x0, x0_b, perf]
        if return_states:
            rvals += [r_policy, r_value]

        return rvals

class Model_inact(Model):
    def __init__(self, modelfile=None, **kwargs):
        Model.__init__(self, modelfile, **kwargs)
    def get_pgi(self, config_or_savefile, seed=1, dt=None, load='best'):
        return PolicyGradient_inact(self.Task, config_or_savefile, seed=seed, dt=dt, load=load)

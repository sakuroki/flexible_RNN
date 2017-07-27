'''
Created on 2016/10/04

@author: sk
'''
import datetime, cPickle, random, os
import numpy as np
import theano

from hf2 import hf_optimizer, SequenceDataset
import generate_model
from generate_dataset import load_dataset

# save setting
#save_dir = None
save_dir = 'path\\to\\save\\directory\\' # anywhere is ok

# load setting
load_model = None

# If model was already generated
#load_model = ''path\\to\\model\\directory\\''

# training setting
load_train_dir = 'path\\to\\tarain_dataset\\directory\\' # specify a path to the dataset
n_train = 40000 # 160000
batch_size_g = 5000
batch_size_cg = 1000
load_valid_dir = 'path\\to\\valid_dataset\\directory\\' # specify a path to the dataset
n_valid = 5000
batch_size_valid = 1000

initial_lambda = .5
n_rep = 50
mu = 1.
patience=5
#temp_save_progress = None
temp_save_progress = 'path\\to\\temp_save\\filename\\' # anywhere is ok
save_log_dir = 'path\\to\\log\\directory\\' # anywhere is ok
log_dir = None

# inputs parameters
discount = 25.
dur_task =750./discount; dur_after = 200./discount
dt = 1.
nt = int(dur_task / dt)

range_offset = [-0.1875, 0.1875 ]
range_stim_onset = [0, 0]
sd_rho_h = 0.1/np.sqrt(discount) #0.1
sd_rho_u = 1./np.sqrt(discount) #1.

# model parameters
nx = 4.; ny = 1.
nh = 100.
tau = 10.

## train process-----

# setting the umber of models to train and the random seed numbers
seeds = [1,2,3,4,5,6,7,8,9,10,11]

# training models
for seed in seeds:
    random.seed(seed)
    print 'Seed: ', seed

    # load or generate RNN model
    if isinstance(load_model, str):
        pkl_file = open(load_model, 'rb')
        model = cPickle.load(pkl_file)
        pkl_file.close()
        p = model[0];
        #p, inputs, s, costs, h, ha, y = generate_model.model_original(nx, nh, ny, p=p, tau = tau)
        p, inputs, s, costs, h, ha, y = generate_model.model_original(nx, nh, ny, p=p, tau = tau)
        print 'load RNN model: ', load_model
    else:
        #p, inputs, s, costs, h, ha, y = model_original(nx, nh, ny, tau = tau)
        p, inputs, s, costs, h, ha, y = generate_model.model_original(nx, nh, ny, tau = tau, seed=seed)

    # laod task input data
    def construct_dataset(dn, n):
        n_file = len(os.listdir(dn))
        id_order = range(n_file); random.shuffle(id_order)
        res_dataset = [[],[],[]]
        for i in xrange(n):
            id = id_order[i]
            ds_file = dn + 'data_' + str(id) + '.ds'
            u, p_T = load_dataset(ds_file)
            rho_h = np.random.normal(0., sd_rho_h, (nt,nh)).astype(theano.config.floatX)

            res_dataset[0].append(u)
            res_dataset[1].append(rho_h)
            res_dataset[2].append(p_T)
            #if i% 1000 == 0: print 'load data: ', i
        return res_dataset

    print 'loading dataset...'
    train = construct_dataset(load_train_dir, n_train) # data for train
    valid = construct_dataset(load_valid_dir, n_valid) # data for valid

    gradient_dataset = SequenceDataset(train, batch_size=None, number_batches=batch_size_g, seed=seed) #5000
    cg_dataset = SequenceDataset(train, batch_size=None, number_batches=batch_size_cg, seed=seed+1234) #1000
    valid_dataset = SequenceDataset(valid, batch_size=None, number_batches=batch_size_valid, seed=seed) #1000

    # training models
    print 'training...'
    time_stamp = datetime.datetime.now()
    log_dir = save_log_dir + time_stamp.strftime('%Y%m%d%H%M%S') +'\\'
    best = hf_optimizer(p, inputs, s, costs, h, ha).train(gradient_dataset, cg_dataset,
                            initial_lambda=initial_lambda, mu=mu, validation=valid_dataset,
                            num_updates=n_rep, patience=patience, save_progress=temp_save_progress,
                            logging = log_dir, max_cg_iterations=50)
    os.remove(temp_save_progress)
    print 'Best model is at iteration ', best[0]

    # test of reconstruction
    #p, inputs, s, costs, h, ha, y = generate_model.model_lossf_only_yT(nx, nh, ny, p = best[2], tau = tau)
    #f_y = theano.function([inputs[0], inputs[1]],y)
    #for i, inp in enumerate(valid_dataset.iterate(update=False)):
    #    print f_y(*inp[0:2])

    # save best the trained model at the best performanced iteration
    if isinstance(save_dir, str):
        p, inputs, s, costs, h, ha, y = generate_model.model_original(nx, nh, ny, p = best[2], tau = tau)
        best_itter = best[0]
        save_model = p, inputs, s, costs, h, ha, y, best_itter
        file_name = save_dir + 'model_seed' + str(seed) + time_stamp.strftime('_%Y%m%d%H%M%S') + '.pkl'
        cPickle.dump(save_model, open(file_name, 'wb'), -1)
        print 'save RNN model: ', file_name

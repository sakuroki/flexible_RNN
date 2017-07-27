'''
Created on 2016/10/04

@author: Satoshi Kuroki

Script for inactivation experiments
'''

import cPickle, random, os, glob
import numpy as np
import scipy
import theano
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import generate_model
from generate_dataset import load_dataset

load_model_dir = 'path\\to\\model\\directory\\' # specify the path to the latest model
load_log_dir = 'path\\to\\log\\directory\\' # specify the path to the log (for initial model setting file)
load_ds_dir = 'path\\to\\dataset\\directory\\' # specify the path to the dataset
save_dir    = 'path\\to\\save\\directory\\' # anywhere is ok
visible_weight_detail = True
visible_result_detail = True
vis_frequency = 1
n_div = 6

# setting values
nx = 4; nh = 100; ny = 1; nt = 30
tau = 10.
n_test = 5000
range_offset = [-0.1875, 0.1875 ]
block = np.linspace(range_offset[0], range_offset[1], n_div+1)
block = (block[:-1] + block[1:]) / 2
block_lab = np.round(block, 2)
ideal_x = np.linspace(range_offset[0], range_offset[1], 1000)

discount = 25.
onset = 0
range_offset = [-0.1875, 0.1875 ]
sd_rho_h = 0.1/np.sqrt(discount) #0.1

beh_legends = ['choice', 'in_m', 'in_c', 'context', 'correct']
inact_step = 10

# graph style
sns.set_style("white"); sns.set_context('talk')

# labeling the data with sensory input offset (mean of sensory input)
def val2label(df, val_tag, min, max, n_div):
    val_lab = val_tag + '_lab'
    spl = n_div + 1
    block = np.linspace(min, max, spl)
    for i in xrange(spl-1):
        temp = df[df[val_tag] >= block[i]]
        temp = temp[temp[val_tag] < block[i+1]]
        temp[val_lab] = (block[i] + block[i+1]) / 2
        if i == 0:
            newdf = temp.copy()
        else:
            newdf = newdf.append(temp)
    return newdf

# plot post-mean (or shuffled) sorted unital order
def plot_W_order(W,ids,save_fn):
    Ws_sort_abs =  W[ids,:]
    Ws_sort_abs =  Ws_sort_abs[:,ids]

    pmax = np.max(np.abs(W.reshape(nh*nh,)))
    plt.pcolor(Ws_sort_abs, vmin = -pmax, vmax = pmax, cmap='jet')
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_label_position('top')
    plt.gca().xaxis.tick_top()
    plt.xlabel('Pre (No.)');plt.ylabel('Post (No.)')
    plt.colorbar(label='Final W Val. - Init. W Val.')
    plt.savefig(save_fn, bbox_inches='tight', dpi=300)
    plt.show()

# function to make models to perform the task
def perform_task(f_y, sort_id, lim, dataset, sort_type):

    print 'performing task:::',sort_type, ' ~', lim
    mod = np.ones((nt,nh)).astype(theano.config.floatX)
    mod[:,sort_id[:lim]] = 0.

    #results_h = np.zeros((n_test, nt, nh))
    results_y = np.zeros((n_test, nt))
    conditions = np.zeros((n_test, 5))

    for i in xrange(n_test):
        u = dataset[i][0]; p_T = dataset[i][1]

        #make inner noise
        rho_h = np.random.normal(0., sd_rho_h, (nt,nh)).astype(theano.config.floatX)

        # calc
        inp = [u, rho_h, mod]
        #res_h = f_h(*inp); results_h[i,:,:] = res_h
        res_y = f_y(*inp); results_y[i,:] = res_y[:,0]

        conditions[i,0] = np.sign(res_y[-1,0])
        conditions[i,1] = np.mean(u[onset:,0]); conditions[i,2] = np.mean(u[onset:,1])
        if u[0,2] == 1.: conditions[i,3] = 1.
        if u[0,3] == 1.: conditions[i,3] = 0.
        conditions[i,4] = 1. if conditions[i,0] == p_T else 0.

    # summarize data
    df = pd.DataFrame(conditions, columns = beh_legends)
    df['sort_type'] = sort_type
    df['n_inact'] = lim
    return df

##### process ########################

# load dataset
n_file = len(os.listdir(load_ds_dir))
id_order = range(n_file); random.shuffle(id_order)
dataset = []
for i in xrange(n_test):
    id = id_order[i]
    ds_file = load_ds_dir + 'data_' + str(id) + '.ds'
    u, p_T = load_dataset(ds_file)
    dataset.append([u, p_T])

# Find model files
allfnames = glob.glob(load_model_dir + 'model_seed*.pkl')

for ff in allfnames:
    print 'File name: ', ff

    seed      = int(ff.split('_')[1][4:])
    timestamp = ff.split('_')[2][:-4]

    if seed == 4: # seed4 model was failured to peform the task
        continue

    # load and set model
    pkl_file = open(ff, 'rb')
    p, inputs, s, costs, h, ha, y, best_ittr = cPickle.load(pkl_file)
    pkl_file.close()

    log_dir = load_log_dir + timestamp + '\\'
    best_model = log_dir + 'model_' + str(best_ittr) + '.pkl'
    init_model = log_dir + 'model_0.pkl'

    # read weight values
    with open(best_model, 'rb') as f:
        model = cPickle.load(f)
    Wrec_last = model[1][1].T
    with open(init_model, 'rb') as f:
        model = cPickle.load(f)
    Wrec_init = model[1][1].T

    # Sort units by post-mean weight changes
    Wrec_diff = Wrec_last - Wrec_init
    Neu_rec_diff = np.mean(np.abs(Wrec_diff), axis = 1)
    desc_id = np.argsort(Neu_rec_diff)[::-1]
    asc_id = np.argsort(Neu_rec_diff)
    shuf_id = range(nh)
    np.random.seed(seed)
    np.random.shuffle(shuf_id)

    # plot and save the sorted weight values
    save_fn = save_dir + 'fig_W_order_descend_seed' + str(seed) + '_' + timestamp +'.png'
    plot_W_order(Wrec_diff, desc_id,save_fn)
    save_fn = save_dir + 'fig_W_order_ascend_seed' + str(seed) + '_' + timestamp +'.png'
    plot_W_order(Wrec_diff, asc_id,save_fn)
    save_fn = save_dir + 'fig_W_order_shuffled_seed' + str(seed) + '_' + timestamp +'.png'
    plot_W_order(Wrec_diff, shuf_id,save_fn)

    # perform task
    p, inputs, s, costs, h, ha, y = generate_model.model_mod_act(nx, nh, ny, p = p, tau = tau)
    f_y = theano.function([inputs[0], inputs[1], inputs[2]],y)
    #f_h = theano.function([inputs[0], inputs[1], inputs[2]],h)

    # full condition
    beh_df = perform_task(f_y, desc_id, 0, dataset, 'full')

    # descending order
    for lim in range(inact_step, nh+1, inact_step):
        temp = perform_task(f_y, desc_id, lim, dataset, 'descend')
        beh_df = pd.concat([beh_df, temp], axis=0)

    # ascending order
    for lim in range(inact_step, nh+1, inact_step):
        temp = perform_task(f_y, asc_id, lim, dataset, 'ascend')
        beh_df = pd.concat([beh_df, temp], axis=0)

    # shuffled order
    for lim in range(inact_step, nh+1, inact_step):
        temp = perform_task(f_y, shuf_id, lim, dataset, 'shuffled')
        beh_df = pd.concat([beh_df, temp], axis=0)

    # grouping and labeling with sensory input offsets
    beh_df = val2label(beh_df,'in_m', range_offset[0], range_offset[1], n_div)
    beh_df = val2label(beh_df,'in_c', range_offset[0], range_offset[1], n_div)

    # add seed and timestamp information
    beh_df['seed'] = seed
    beh_df['timestamp'] = timestamp

    # save dataframe
    save_fn = save_dir + 'beh_df_seed' + str(seed) + '.pkl'
    cPickle.dump(beh_df, open(save_fn, 'wb'), -1)

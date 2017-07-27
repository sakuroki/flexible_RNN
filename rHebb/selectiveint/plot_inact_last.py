# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 17:08:31 2017

@author: sasak
"""

from pylab import *
import glob, os
import pandas as pd
import seaborn as sns
import numpy as np
import cPickle as pickle
import csv

data_dir = 'C:\\Users\\sasak\\OneDrive\\workspace\\BiologicallyPlausibleLearningRNN\\selectiveint\\test\\'
os.chdir(data_dir)

data_fn = 'df_alldata.pkl'

#Interactive mode 
ion()

fgr = figure()
fgr.set_size_inches(4, 4)
fgr.set_facecolor('white')

font = {#'family' : 'normal',
#                'weight' : 'bold',
                       'size'   : 8}

rc('font', **font)

if not os.path.isfile(data_fn):
    vals=[]
    #allfnames = glob.glob('rs_*type*.txt') 
    allfnames = glob.glob('lastr_*type*.txt') 
    # allfnames = allfnames[:100] # process for short test
    for ff in allfnames:
        # print ff
        z = loadtxt(ff);
        #vals.append(z[-1,0])
        vals.append(z)
    
    print "Files read!"
    
    #print allfnames[0].split('_')[9][1:]
    
    allbias0s  = pd.Series([float(ss.split('_')[3]) for ss in allfnames])
    allbias1s  = pd.Series([float(ss.split('_')[5]) for ss in allfnames])
    trialtypes = pd.Series([int(ss[10]) for ss in allfnames])
    trialnums  = pd.Series([int(ss.split('_')[6]) for ss in allfnames])
    seednums   = pd.Series([int(ss.split('_')[7][7:]) for ss in allfnames])
    inactnums  = pd.Series([int(ss.split('_')[9][1:]) for ss in allfnames])
    inactids   = pd.Series(['_'.join(ss.split('_')[10:])[:-4] for ss in allfnames])
                
    #mat1 = vstack((trialtypes, allbias1s, allbias2s, trialnums, inactnums, inactid, vals)).T
                 
    mat = pd.DataFrame({
                         'trial_type':trialtypes,
                         'bias0':allbias0s,
                         'bias1':allbias1s,
                         'trial_num':trialnums,
                         'seed_num' :seednums,
                         'inact_num':inactnums,
                         'inact_id':inactids,
                         'value':vals})
    
    # identify sort types
    mat.loc[mat[mat["inact_id"]=="full"].index,"sort_type"] = "full"
    
    Jdata_dir = 'C:\\Users\\sasak\\OneDrive\\workspace\\BiologicallyPlausibleLearningRNN\\selectiveint\\test_20170208\\'
    n_neuron = 200
    def trans_list(id_,target_):
            for i, id in enumerate(id_):
                if id == target_:
                    temp_id = id_[i]
                    id_ = np.delete(id_, i)
                    id_ = np.append(id_, temp_id)
                    return id_
                    
    for RNGSEED in range(1,21):
        fn = Jdata_dir + 'J_RNGSEED'+str(RNGSEED)+'.txt'
        Wrec_last = np.zeros((n_neuron, n_neuron))
        f = open(fn, 'r')
        reader = csv.reader(f, delimiter=" ")
        for i, row in enumerate(reader):
            roww = [float(r) for r in row if r != '']
            Wrec_last[i,:] = roww
        f.close()
        
        Neu_rec_diff = np.mean(np.abs(Wrec_last), axis = 1)
        
        sort_id = np.argsort(Neu_rec_diff)[::-1]
        sort_id = trans_list(sort_id,1)
        sort_id = trans_list(sort_id,10)
        sort_id = trans_list(sort_id,11)
        sort_id = trans_list(sort_id,0)
        descend_id = str(sort_id[0])
        for sid in sort_id[1:5]:
            descend_id = descend_id + '_' + str(sid)
        
        sort_id = np.argsort(Neu_rec_diff)
        sort_id = trans_list(sort_id,1)
        sort_id = trans_list(sort_id,10)
        sort_id = trans_list(sort_id,11)
        sort_id = trans_list(sort_id,0)
        ascend_id = str(sort_id[0])
        for sid in sort_id[1:5]:
            ascend_id = ascend_id + '_' + str(sid)
        
        sort_id = range(n_neuron)
        np.random.seed(RNGSEED)
        np.random.shuffle(sort_id)
        sort_id = trans_list(sort_id,1)
        sort_id = trans_list(sort_id,10)
        sort_id = trans_list(sort_id,11)
        sort_id = trans_list(sort_id,0)
        shuffled_id = str(sort_id[0])
        for sid in sort_id[1:5]:
            shuffled_id = shuffled_id + '_' + str(sid)
            
        mat.loc[mat[mat["inact_id"]==descend_id].index,"sort_type"] = "descend"
        mat.loc[mat[mat["inact_id"]==ascend_id].index,"sort_type"] = "ascend"
        mat.loc[mat[mat["inact_id"]==shuffled_id].index,"sort_type"] = "shuffled"
        
    
    # calc accuracy
    mat    = mat[(mat['inact_num'] <= 100)]
    mat_c0 = mat[(mat['trial_type'] == 0)]
    mat_c0 = mat_c0[(mat_c0['bias0'] != 0)]
    mat_c1 = mat[(mat['trial_type'] == 1)]
    mat_c1 = mat_c1[(mat_c1['bias1'] != 0)]
                    
    mat_c0['accuracy'] = (np.sign(mat_c0.loc[:,'bias0']) == np.sign(mat_c0.loc[:,'value'])).astype(int)
    mat_c1['accuracy'] = (np.sign(mat_c1.loc[:,'bias1']) == np.sign(mat_c1.loc[:,'value'])).astype(int)
    
    mat = pd.concat([mat_c0, mat_c1],axis = 0)
    
    data_fn = 'df_alldata.pkl'
    with open(data_fn, mode='wb') as f:
        pickle.dump(mat, f)

else:
    with open(data_fn, mode='rb') as f:
        mat = pickle.load(f)
#print mat
#print mat.isnull().any()

sns.set_style('white');# sns.set_context("poster")
sns.pointplot(x="inact_num", y="accuracy", hue="sort_type", data=mat)
fig_name = 'figure_result_plot_inact.png' 
savefig(fig_name, bbox_inches='tight', dpi=300)
show()
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:52:26 2017

@author: sasak
"""

import subprocess
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

n_neuron = 200
id_bin = 10; id_ran = 10

save_dir = 'test'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
os.chdir(save_dir)

for RNGSEED in range(1,21):
    fn = 'C:\\Users\\sasak\\OneDrive\\workspace\\BiologicallyPlausibleLearningRNN\\selectiveint\\test_20170208\\J_RNGSEED'+str(RNGSEED)+'.txt'
    Wrec_last = np.zeros((n_neuron, n_neuron))
    
    f = open(fn, 'r')
    reader = csv.reader(f, delimiter=" ")
    for i, row in enumerate(reader):
        roww = [float(r) for r in row if r != '']
        Wrec_last[i,:] = roww
    f.close()
    
    # performing task with full Ws
    i=0
    for BIAS1 in  [-.5, -.4, -.3, -.2, -.1, 0.0, .1,  .2, .3, .4, .5]:
        for BIAS2 in  [-.5, -.4, -.3, -.2, -.1, 0.0, .1, .2, .3, .4, .5]:
            i = i+1
            print i
            mycmd="./net_inact TEST  RNGSEED "+str(RNGSEED)+" ETA .01 ALPHAMODUL 30.0 MAXDW 2e-4 ALPHABIAS .5 BIAS1 " + str(BIAS1)+ " BIAS2 "+str(BIAS2) # Note that PROBAMODUL is not set to zero - perturbations even in test!
            #mycmd="./net_inact TEST  ETA .01 ALPHAMODUL 30.0 MAXDW 2e-4 ALPHABIAS .5 BIAS1 " + str(BIAS1)+ " BIAS2 "+str(BIAS2) + " IDINACT 199 180 2 4 5 6 7 8 9 10 11"
            print mycmd
            returncode = subprocess.call(mycmd)
    
    Neu_rec_diff = np.mean(np.abs(Wrec_last), axis = 1)
    sorted_val =  np.sort(Neu_rec_diff)[::-1]
    sort_id = np.argsort(Neu_rec_diff)[::-1]
    
    def trans_list(val_,id_,target_):
        for i, id in enumerate(id_):
            if id == target_:
                temp_v = val_[i]
                temp_id = id_[i]
                val_ = np.delete(val_, i)
                id_ = np.delete(id_, i)
                val_ = np.append(val_, temp_v)
                id_ = np.append(id_, temp_id)
                return val_, id_
    
    sorted_val, sort_id = trans_list(sorted_val,sort_id,1)
    sorted_val, sort_id = trans_list(sorted_val,sort_id,10)
    sorted_val, sort_id = trans_list(sorted_val,sort_id,11)
    sorted_val, sort_id = trans_list(sorted_val,sort_id,0)
    
    Ws_sort_abs =  Wrec_last[sort_id,:]
    Ws_sort_abs =  Ws_sort_abs[:,sort_id]
    pmax = np.max(np.abs(Wrec_last.reshape(40000,)))
    plt.pcolormesh(Ws_sort_abs,vmin = -pmax, vmax = pmax,cmap='jet')
    plt.gca().invert_yaxis()
    plt.xlabel('Pre');plt.ylabel('Post')
    plt.colorbar(label='Final W Val. - Init. W Val.')
    plt.gca().xaxis.set_label_position('top') 
    plt.gca().xaxis.tick_top()
    fig_name = 'figure_inact_Wrec_descending_RNGSEED' +str(RNGSEED) + '.png' 
    plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.close()
    
    for bin in range(id_bin):
        
        str_idinact = " IDINACT"
        for id in sort_id[:(bin+1)*id_ran]:
            str_idinact = str_idinact + " " + str(id)
            
        for BIAS1 in  [-.5, -.4, -.3, -.2, -.1, 0.0, .1,  .2, .3, .4, .5]:
            for BIAS2 in  [-.5, -.4, -.3, -.2, -.1, 0.0, .1, .2, .3, .4, .5]:
                i = i+1
                print i
                mycmd="./net_inact TEST  RNGSEED "+str(RNGSEED)+" ETA .01 ALPHAMODUL 30.0 MAXDW 2e-4 ALPHABIAS .5 BIAS1 " + str(BIAS1)+ " BIAS2 "+str(BIAS2) + str_idinact  # Note that PROBAMODUL is not set to zero - perturbations even in test!
                #mycmd="./net_inact TEST  ETA .01 ALPHAMODUL 30.0 MAXDW 2e-4 ALPHABIAS .5 BIAS1 " + str(BIAS1)+ " BIAS2 "+str(BIAS2) + " IDINACT 199 180 2 4 5 6 7 8 9 10 11"
                print mycmd
                returncode = subprocess.call(mycmd)
          
                
    # ascending order
    sorted_val =  np.sort(Neu_rec_diff)
    sort_id = np.argsort(Neu_rec_diff)
    
    sorted_val, sort_id = trans_list(sorted_val,sort_id,1)
    sorted_val, sort_id = trans_list(sorted_val,sort_id,10)
    sorted_val, sort_id = trans_list(sorted_val,sort_id,11)
    sorted_val, sort_id = trans_list(sorted_val,sort_id,0)
    
    Ws_sort_abs =  Wrec_last[sort_id,:]
    Ws_sort_abs =  Ws_sort_abs[:,sort_id]
    pmax = np.max(np.abs(Wrec_last.reshape(40000,)))
    plt.pcolormesh(Ws_sort_abs,vmin = -pmax, vmax = pmax,cmap='jet')
    plt.gca().invert_yaxis()
    plt.xlabel('Pre');plt.ylabel('Post')
    plt.colorbar(label='Final W Val. - Init. W Val.')
    plt.gca().xaxis.set_label_position('top') 
    plt.gca().xaxis.tick_top()
    fig_name = 'figure_inact_Wrec_asceinding_RNGSEED' +str(RNGSEED) + '.png' 
    plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.close()
    
    for bin in range(id_bin):
        
        str_idinact = " IDINACT"
        for id in sort_id[:(bin+1)*id_ran]:
            str_idinact = str_idinact + " " + str(id)
            
        for BIAS1 in  [-.5, -.4, -.3, -.2, -.1, 0.0, .1,  .2, .3, .4, .5]:
            for BIAS2 in  [-.5, -.4, -.3, -.2, -.1, 0.0, .1, .2, .3, .4, .5]:
                i = i+1
                print i
                mycmd="./net_inact TEST  RNGSEED "+str(RNGSEED)+" ETA .01 ALPHAMODUL 30.0 MAXDW 2e-4 ALPHABIAS .5 BIAS1 " + str(BIAS1)+ " BIAS2 "+str(BIAS2) + str_idinact  # Note that PROBAMODUL is not set to zero - perturbations even in test!
                #mycmd="./net_inact TEST  ETA .01 ALPHAMODUL 30.0 MAXDW 2e-4 ALPHABIAS .5 BIAS1 " + str(BIAS1)+ " BIAS2 "+str(BIAS2) + " IDINACT 199 180 2 4 5 6 7 8 9 10 11"
                print mycmd
                returncode = subprocess.call(mycmd)
                
    # shuffle
    sort_id = range(n_neuron)
    np.random.seed(RNGSEED)
    np.random.shuffle(sort_id)
    sorted_val = Neu_rec_diff[sort_id]
    
    sorted_val, sort_id = trans_list(sorted_val,sort_id,1)
    sorted_val, sort_id = trans_list(sorted_val,sort_id,10)
    sorted_val, sort_id = trans_list(sorted_val,sort_id,11)
    sorted_val, sort_id = trans_list(sorted_val,sort_id,0)
    
    Ws_sort_abs =  Wrec_last[sort_id,:]
    Ws_sort_abs =  Ws_sort_abs[:,sort_id]
    pmax = np.max(np.abs(Wrec_last.reshape(40000,)))
    plt.pcolormesh(Ws_sort_abs,vmin = -pmax, vmax = pmax,cmap='jet')
    plt.gca().invert_yaxis()
    plt.xlabel('Pre');plt.ylabel('Post')
    plt.colorbar(label='Final W Val. - Init. W Val.')
    plt.gca().xaxis.set_label_position('top') 
    plt.gca().xaxis.tick_top()
    fig_name = 'figure_inact_Wrec_shuffle_RNGSEED' +str(RNGSEED) + '.png' 
    plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.close()
    
    for bin in range(id_bin):
        
        str_idinact = " IDINACT"
        for id in sort_id[:(bin+1)*id_ran]:
            str_idinact = str_idinact + " " + str(id)
            
        for BIAS1 in  [-.5, -.4, -.3, -.2, -.1, 0.0, .1,  .2, .3, .4, .5]:
            for BIAS2 in  [-.5, -.4, -.3, -.2, -.1, 0.0, .1, .2, .3, .4, .5]:
                i = i+1
                print i
                mycmd="./net_inact TEST  RNGSEED "+str(RNGSEED)+" ETA .01 ALPHAMODUL 30.0 MAXDW 2e-4 ALPHABIAS .5 BIAS1 " + str(BIAS1)+ " BIAS2 "+str(BIAS2) + str_idinact  # Note that PROBAMODUL is not set to zero - perturbations even in test!
                #mycmd="./net_inact TEST  ETA .01 ALPHAMODUL 30.0 MAXDW 2e-4 ALPHABIAS .5 BIAS1 " + str(BIAS1)+ " BIAS2 "+str(BIAS2) + " IDINACT 199 180 2 4 5 6 7 8 9 10 11"
                print mycmd
                returncode = subprocess.call(mycmd)
print 'Finish Process'
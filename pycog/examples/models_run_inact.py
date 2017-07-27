# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 00:20:51 2017

@author: Satoshi Kuroki

Script to perform tasks with multiple models for inactivation experiment
"""

import os, shutil, subprocess, glob

work_dir   = 'path\\to\\workdir'
load_p_dir = work_dir + '\\work\\data\\mante\\'
save_p_dir = 'path\\to\\save\\dir\\'
last_file = 'mante.pkl'
init_file = 'mante_init.pkl'

os.chdir(work_dir)
seeds = [100,2,3,4,5,6,7,8,9,10,11] # set seeds
for seed in seeds:

    # load models
    load_dir = load_p_dir + 'test_seed' + str(seed) + '\\'
    print(load_dir)
    shutil.copy(load_dir + last_file, load_p_dir)
    shutil.copy(load_dir + init_file, load_p_dir)

    # run script to perform the task
    cmd = ['python','do.py','models/mante','run','analysis/mante','inactivation']
    print(cmd)
    returncode = subprocess.call(cmd)

    # save results
    allfn = glob.glob(save_p_dir + 'mante_trials_inact*.pkl')
    save_dir = save_p_dir + 'test_seed' + str(seed)
    print(save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for ff in allfn:
        shutil.move(ff, save_dir)

    # discard temporal files
    os.remove(load_p_dir + last_file)
    os.remove(load_p_dir + init_file)

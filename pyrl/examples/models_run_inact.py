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
init_file = 'mante_0.pkl'

os.chdir(work_dir)
seeds = [100,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] # set seeds
for seed in seeds:

    # laod models
    load_dir = load_p_dir + 'test_seed' + str(seed) + '\\'
    print(load_dir)
    shutil.copy(load_dir + last_file, load_p_dir)
    shutil.copy(load_dir + init_file, load_p_dir)

    # run models to perform tasks
    cmd = ['python','do.py','models/mante','run','analysis/mante','inactivation_diff','100']
    print(cmd)
    returncode = subprocess.call(cmd)

    # save (and move) the results
    allfn = glob.glob(save_p_dir + 'trials_behavior_inactinact*.pkl')
    save_dir = save_p_dir + 'test_seed' + str(seed)
    print(save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for ff in allfn:
        shutil.move(ff, save_dir)

    os.remove(load_p_dir + last_file)
    os.remove(load_p_dir + init_file)

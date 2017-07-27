# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 00:20:51 2017

@author: Satoshi Kuroki

Script for generate multiple models for inactivation experiment
"""

import os, shutil, subprocess, glob

work_dir   = 'path\\to\\workdir'
save_p_dir = work_dir + '\\work\\data\\mante\\'


os.chdir(work_dir)
seeds = [100,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] # set random seeds

# generate models
for seed in seeds:
    cmd = ['python','do.py','--seed',str(seed),'models/mante','train']
    print(cmd)
    returncode = subprocess.call(cmd)

    save_dir = save_p_dir + 'test_seed' + str(seed)
    allfn = glob.glob(save_p_dir + 'mante*.pkl')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for ff in allfn:
        shutil.move(ff, save_dir)

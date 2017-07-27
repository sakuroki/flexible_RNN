'''
Created on 2016/09/27

@author: Satoshi Kuroki

Script for generate datasets of context-dependent integration task
'''

import datetime, cPickle, random, os
import numpy as np
import matplotlib.pyplot as plt
import theano

def generate_dataset(save_dir, session_label = ['train', 'valid', 'test'], n_session = [10000, 1000, 1000],
                     dur_task = 750., dt = 1.,range_offset = [-0.1875, 0.1875 ],sd_rho_u = 1,
                     range_stim_onset = [0, 0], task_type = 'Contextual'):
'''
generate_dataset::: function to generate dataset

Input
save_dir:         Path to save directory of the generated task data.
session_label:    labels of dataset. (default=['train', 'valid', 'test'])
n_session:        The number of data of each labeled dataset. (default=[10000, 1000, 1000])
dur_task:         The total number of time steps. (default=750)
dt:               Size of Time step. (default=1)
range_offset:     min and max values of sensory offset value. (default = [-0.1875, 0.1875 ])
sd_rho_u:         The standard value of gaussian noise of sensory input. (default=1)
range_stim_onset: The range of sensory input onset timing (default=[0, 0])
task_type:        The type of the test (now we set only for 'Contextual')

Save
info_str: Meta information of the generated dataset.
u:        Sensory and contexual time series inputs.
p_T:      Teach signals of choices.
'''

        # make directory
        time_stamp = datetime.datetime.now()
        dset_name = 'dataset' + time_stamp.strftime('_%Y%m%d%H%M%S')
        save_dir = save_dir + dset_name + '/'
        os.mkdir(save_dir)

        # save meta_info
        info_str = 'Dataset name: ' + dset_name + '\n'
        info_str = info_str + 'Date: ' + time_stamp.strftime('%Y/%m/%d %H:%M:%S') + '\n'
        info_str = info_str + 'Created by: sk\n\n'
        for (S, type) in zip(n_session, session_label):
            info_str = info_str + 'Number of ' + type + ' data: ' + str(S) + '\n'
        info_str = info_str + 'Task duration (ms): ' + str(dur_task) + '\n'
        info_str = info_str + 'dt (ms): ' + str(dt) + '\n'
        info_str = info_str + 'Signal offset range: ' + str(range_offset) + '\n'
        info_str = info_str + 'Std of noise: ' + str(sd_rho_u) + '\n'
        info_str = info_str + 'Stim onset range (ms): ' + str(range_stim_onset) +'\n\n'
        info_str = info_str + 'Task type: ' + task_type + '\n'
        info_str = info_str + 'u[0]: Motor input\n'
        info_str = info_str + 'u[1]: Color input\n'
        info_str = info_str + 'u[2]: Context motor\n'
        info_str = info_str + 'u[3]: Context color\n'
        info_str = info_str + 'p_T: Teach signal\n'

        file_name = save_dir + 'Info.txt'
        f = open(file_name, 'wb')
        f.write(info_str)
        f.close()
        print 'save Info: ', file_name

        # generate task datasets
        for (S, type) in zip(n_session, session_label):

            # make directory
            os.mkdir(save_dir + type)

            # make context
            N_t = dur_task / dt
            c_array = np.random.randint(0, 2, S)

            # make input data
            for i in xrange(0, int(S)):
                # set input
                d_m = np.random.uniform(range_offset[0],range_offset[1]) * np.ones(N_t)
                d_c = np.random.uniform(range_offset[0],range_offset[1]) * np.ones(N_t)
                rho_m = np.random.normal(0., sd_rho_u, N_t)
                rho_c = np.random.normal(0., sd_rho_u, N_t)
                u_m = d_m + rho_m
                u_c = d_c + rho_c

                # set zero until stim onset
                stim_onset = random.randint(range_stim_onset[0],range_stim_onset[1])
                u_m[0:stim_onset] = 0; u_c[0:stim_onset] = 0


                # set context dependent information
                context = c_array[i]
                if context == 0:
                    u_cm = np.ones(N_t); u_cc = np.zeros(N_t)
                    p_T = 1. if d_m[0] >= 0 else -1.
                elif context == 1:
                    u_cm = np.zeros(N_t); u_cc = np.ones(N_t)
                    p_T = 1. if d_c[0] >= 0 else -1.

                u = np.c_[u_m, u_c, u_cm, u_cc].astype(theano.config.floatX)

                save_data = u, p_T
                file_name = save_dir + type + '/data_' + str(i) + '.ds'
                cPickle.dump(save_data, open(file_name, 'wb'), -1)
                if i% 100 == 0: print 'save data: ', file_name

def load_dataset(filename):
'''
load_dataset::: Function to load the generated dataset

Output
u:   Sensory and contexual time series inputs.
p_T: Teach signals of choices.
'''
    pkl_file = open(filename, 'rb')
    u, p_T = cPickle.load(pkl_file)
    pkl_file.close()
    return u, p_T

def check_dataset(load_dir, n_data = 30, start_id = 0, range_offset = [-0.1875, 0.1875 ]):
'''
check_dataset::: function to visualize the dataset

Input
load_dir:     Path to datasets to check
n_data:       The number of dataset to visualize (default=30)
start_id:     The  no. of first data to visualize (default=0)
range_offset: Min and max of sensory input offset to set the y-axis.
              (defualt=[-0.1875, 0.1875])

'''

    for i in xrange(n_data):
        file_name = load_dir + 'data_' + str(i + start_id) + '.ds'
        pkl_file = open(file_name, 'rb')
        dataset = cPickle.load(pkl_file)
        pkl_file.close()
        u = dataset[0]
        p_T = dataset[1]
        print 'load data: ', file_name

        tsig = np.ones(u.shape[0]) * p_T * (range_offset[1]/2.)
        base = np.zeros(u.shape[0])

        plt.subplot(2,1,1)
        plt.plot(u[:,0], label='Motion', color='blue')
        plt.plot(u[:,1], label='Color', color='red')
        plt.plot(tsig, label='T. Sig', color='yellow')
        plt.plot(base, label='Base', color='black')
        plt.ylim(range_offset[0] * 1.1, range_offset[1] * 1.1)
        plt.title('Input'); plt.legend()

        plt.subplot(2,1,2)
        plt.plot(u[:,2], label='Cont. Motion', color='blue')
        plt.plot(u[:,3], label='Cont. Color', color='red')
        plt.ylim(-1.1, 1.1)
        plt.title('Context'); plt.legend()

        plt.show()

#save_dir = 'path/to/dataset'
#generate_dataset(save_dir, session_label = ['train', 'valid', 'test'], n_session = [40000, 5000, 5000],
#                     dur_task = 750./25., dt = 1.,range_offset = [-0.1875, 0.1875 ],sd_rho_u = 1./5.,
#                     range_stim_onset = [0, 0], task_type = 'Contextual')

# check data setting
#load_dir = 'path/to/dataset'
#check_dataset(load_dir, n_data=30, start_id=0,range_offset = [-0.1875, 0.1875 ])

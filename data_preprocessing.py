#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


# In[2]:


import os, json
from tqdm import tqdm


# In[3]:


from transforms import FITPS


# In[5]:


def where_runtimes(x: np.ndarray, thresh: float = 0.01,) -> np.ndarray:
    
    m = np.mean(np.abs(x), axis=-1)
    
    mask = m > thresh
        
    b = np.where(mask, 1, 0)
            
    derivative = np.diff(b, prepend=0, append=0)
    mask = derivative != .0
    t = np.argwhere(mask)
    t = t.reshape(-1, 2)
    t[:, 1] -= 1
        
    return t
    


# In[6]:


def where_jumps(x):
#     plt.plot(x.ravel())
    x = np.std(x, axis=1)
#     d = x[1:] - x[:-1]
    d = np.diff(x)
#     plt.plot(np.abs(d))

    return np.argwhere(np.abs(d)>0.01)


# # Processing of aggregated data

# In[7]:


data_meta = pd.read_json("data/metadata_aggregated.json").to_dict('dict')


# In[8]:


# On all data 
freq_list = [data_meta[idx]['header']['sampling_frequency'][:-2] for idx in range(1,len(data_meta))]
data = []
sets_considered = 0
sets_total = 0

for set_num in tqdm(range(1,576,1)):  
#     print(set_num)
    if_one = False
    sets_total += 1
    
    # Aggregate data
    data_agg = pd.read_csv(f"data/aggregated/{set_num}.csv", names=["Current", "Voltage"])

    data_i_all = data_agg["Current"].values
    data_u_all = data_agg["Voltage"].values
    
    # Meta data
    dt = data_meta[set_num]['appliances']

    time_list = []
    devices_list = []

    for idx, apl in enumerate(dt):
        on = apl['on'].replace("[", "").replace("]", "")
        off = apl['off'].replace("[", "").replace("]", "")
        on = on.split()
        off = off.split()
        if (len(on) > len(off)):
            off.append(str(len(data_i_all)))
        for idx_switch in range(len(on)):
            on_cur = int(on[idx_switch])
            off_cur = int(off[idx_switch])
            time_list.append( [on_cur, off_cur] )
            devices_list.append(apl["type"])

    fitps = FITPS(60)
    data_u_all, data_i_all = fitps(data_u_all, data_i_all, freq_list[0])

    jumps_list = where_jumps(data_i_all)
    jumps_list = [jump[0] for jump in jumps_list]

    # Period is 500 points (frequence / 60)
    for idx1 in range(0, len(jumps_list)-1, 1):
        jump_left = jumps_list[idx1]
        jump_right = jumps_list[idx1+1] 
        periods_cur = jump_right - jump_left
        if periods_cur > 10:
            period_num = periods_cur // 10

            for period in range(period_num):
                time_left = (jump_left + 10*period)*500
                time_right = (jump_left + 10*(period + 1))*500

                data_i = data_agg["Current"][time_left:time_right].values
                data_u = data_agg["Voltage"][time_left:time_right].values

                fitps = FITPS(60)
                data_u, data_i = fitps(data_u, data_i, freq_list[0])
  
                devices_cur = []
                for idx2, times_bound in enumerate(time_list):
                    if (times_bound[0] < time_left < times_bound[1]):
                        if (times_bound[0] < time_right < times_bound[1]):
                            devices_cur.append(devices_list[idx2])
                
                if (len(devices_cur) > 0 ):
                    if 0:
                        print(devices_cur)
                        plt.plot(data_i.ravel())
                        plt.show()
                        
                    if_one = True
                    data.append((data_i, devices_cur))
                    
    if if_one:
        sets_considered += 1

        
print("Datasets were considered out of total: {} / {}".format(sets_considered, sets_total) )
                
# np.save('data/data_aggregate_split.npy', data, allow_pickle=True)


# # Processing of submetered data

# In[9]:


data_meta = pd.read_json("data/metadata_submetered.json").to_dict('dict')


# In[10]:


# On all data 
freq_list = [data_meta[idx]['header']['sampling_frequency'][:-2] for idx in range(1,len(data_meta))]
data = []
sets_considered = 0
sets_total = 0

for set_num in tqdm(range(1800,1877,1)):
#     print(set_num)
    if_one = False
    sets_total += 1
    
    # Aggregate data
    data_sub = pd.read_csv(f"data/submetered/{set_num}.csv", names=["Current", "Voltage"])

    data_i_all = data_sub["Current"].values
    data_u_all = data_sub["Voltage"].values
    
    # Meta data
    dt = data_meta[set_num]['appliance']
    
    try:
        device_cur = dt["type"]
    except:
        device_cur = None 
    
    fitps = FITPS(60)
    data_u_all, data_i_all = fitps(data_u_all, data_i_all, freq_list[0])

    jumps_list = where_jumps(data_i_all)
    jumps_list =  [0] + [jump[0] for jump in jumps_list]
    jumps_list.append(len(data_i_all))

    # Period is 500 points (frequence / 60)
    for idx1 in range(0, len(jumps_list)-1, 1):
        jump_left = jumps_list[idx1]
        jump_right = jumps_list[idx1+1] 
        periods_cur = jump_right - jump_left
        if periods_cur > 10:
            period_num = periods_cur // 10

            for period in range(period_num):
                time_left = (jump_left + 10*period)*500
                time_right = (jump_left + 10*(period + 1))*500

                data_i = data_sub["Current"][time_left:time_right].values
                data_u = data_sub["Voltage"][time_left:time_right].values
                
                means_list = []
                for i in range(10):
                    means_list.append(np.abs(data_i[500*i:500*(i+1)]).mean())
                    
                mean_cur = abs( np.array(means_list).mean() ) 
                mark_mean = True
                for i in range(10):
                    ratio_cur = abs(means_list[i]) / mean_cur
                    if (ratio_cur > 5) or (ratio_cur < 1.0/5.0):
                        mark_mean = False
                        break

                fitps = FITPS(60)
                data_u, data_i = fitps(data_u, data_i, freq_list[0])
                                  
                if device_cur and (mark_mean) and (np.amax(np.abs(data_i))>0.1): #threshhold to remove noises
                    if 0:
                        print(device_cur)
                        plt.plot(data_i.ravel())
                        plt.show()
                    
                    if_one = True
                    data.append((data_i, device_cur))

    if if_one:
        sets_considered += 1
        
print("Appliances were considered out of total: {} / {}".format(sets_considered, sets_total) )
                    
    
# np.save('data/data_submetered_split.npy', data, allow_pickle=True)


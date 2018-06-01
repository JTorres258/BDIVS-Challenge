# -*- coding: utf-8 -*-
"""
Created on Thu May 31 16:32:51 2018

@author: BDIVS
"""


import subprocess

i = 1
k_w = (24, 20, 16, 12)
k_h = (18, 15, 12, 9)
RP_filters = (64, 32, 16, 8)

for j in range (4):
    for k in range (4):
    
        subprocess.call('python train_custom_resnet.py --experiment_rootdir=./models/'+str(i)+' --train_dir="./data_parking/training" --val_dir="./data_parking/validation" --img_mode=rgb --rn_version=1 --n_layers=1 --batch_size=4  --img_width=800 --img_height=600 --kernel_width='+str(k_w[j])+' --kernel_height='+str(k_h[j])+' --RP_filters='+str(RP_filters[k]), shell = True)
        i += 1
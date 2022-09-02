#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 10:53:25 2020

@author: krishna
"""

import os
import numpy as np
import argparse
import glob



def create_meta(folder_path,store_loc,mode='train'):
    if not os.path.exists(store_loc):
        os.makedirs(store_loc)
    
    if mode=='train':
        root_dir = os.path.join(folder_path,'TRAIN')
        all_files = sorted(glob.glob(root_dir+'/*.json'))
        meta_store = store_loc+'/training.txt'
        fid = open(meta_store,'w')
        for filepath in all_files:
            fid.write(filepath+'\n')
        fid.close()
    elif mode=='test':
        root_dir = os.path.join(folder_path,'TEST')
        all_files = sorted(glob.glob(root_dir+'/*.json'))
        meta_store = store_loc+'/testing.txt'
        fid = open(meta_store,'w')
        for filepath in all_files:
            fid.write(filepath+'\n')
        fid.close()
    else:
        print('Error in creating meta files')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Configuration for data preparation")
    parser.add_argument("--processed_data", default="/media/newhd/ASR_datasets/TIMIT/processed_data", type=str,help='Dataset path')
    parser.add_argument("--meta_store_path", default="data_utils/", type=str,help='Save directory after processing')
    config = parser.parse_args()
    create_meta(config.processed_data,config.meta_store_path,mode='train')
    create_meta(config.processed_data,config.meta_store_path,mode='test')
    
    




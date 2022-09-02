#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 12:16:28 2020

@author: krishna
"""


import numpy as np
import torch
from utils import utility
from utils.FeatureLoader import Feature_loader

class SpeechDataGenerator():
    """Speech dataset."""

    def __init__(self, manifest,max_len,pad_token):
        """
        Read the textfile and get the paths
        """
        self.max_len = max_len
        self.pad_id = pad_token
        self.json_links = [line.rstrip('\n').split(' ')[0] for line in open(manifest)]
        
    def __len__(self):
        return len(self.json_links)
        
    
    def __getitem__(self, idx):#获取每一个样本数据
        json_link =self.json_links[idx]
        featureloader = Feature_loader(json_link)
        norm_spec, char_map_list,char_list = featureloader.load_dataset()
        #lang_label=lang_id[self.audio_links[idx].split('/')[-2]]
        batch = {'features': torch.from_numpy(np.ascontiguousarray(norm_spec)), 
                  'char_map_seq': torch.from_numpy(np.ascontiguousarray(char_map_list)),
                  'chars': char_list,
                  'max_len':self.max_len,
                  'pad_token':self.pad_id}
        return batch
    
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:15:31 2020

@author: krishna
"""

import os
import numpy as np
import json
import torch
import librosa


def feature_stack(feature):
    feat_dim,time=feature.shape
    stacked_feats=[]
    for i in range(0,time-3,3):
        splice = feature[:,i:i+3]
        stacked_feats.append(np.array(splice).flatten())
    return np.asarray(stacked_feats)


def SpecAugment(stacked_feature):
    time,feat_dim=stacked_feature.shape
    ##### Masking 5% of the data
    win_len = round(time*0.05)
    mask_start_index = np.random.randint(0, time-win_len)
    create_zero_mat = np.zeros((win_len,feat_dim))
    stacked_feature[mask_start_index:mask_start_index+win_len,:] = create_zero_mat
    masked_features = stacked_feature
    return masked_features


def pad_labels(labels,pad_token,max_len):
    #labels = labels.int()
    input_len=len(labels)
    if input_len<max_len:    
        pad_len=max_len-input_len
        pad_seq = torch.fill_(torch.zeros(pad_len), pad_token).int()
        labels = torch.cat((labels,pad_seq))
    return labels



def pad_sequence_feats(features_list):
    lengths =[feature_mat.shape[0]  for feature_mat in features_list]
    max_length = max(lengths)
    padded_feat_batch=[]
    for feature_mat in features_list:
        pad_mat = torch.zeros((max_length-feature_mat.shape[0],feature_mat.shape[1]))
        padded_feature = torch.cat((feature_mat,pad_mat),0)
        padded_feat_batch.append(padded_feature.T)
    return padded_feat_batch




def speech_collate(batch):
    features=[]
    input_length_lengths=[]
    pad_targets = []
    org_text = []
    for item in batch:
        features.append(item['features'])
        input_length_lengths.append(int(item['features'].shape[0]))
        org_text.append(item['chars'])
        targets = pad_labels((item['char_map_seq']),int(item['pad_token']),int(item['max_len']))
        pad_targets.append(targets)
        
    padded_feats = pad_sequence_feats(features)
    
    #masked_feats_padded = pad_sequence_feats(masked_feats)
    #gt_feats_padded = pad_sequence_feats(gt_feats)
    
    return padded_feats,input_length_lengths, pad_targets,org_text




def create_vocab_dict(vocab_path):
    vocab_list = [line.rstrip('\n') for line in open(vocab_path)]
    i=0
    vocab_dict={}
    for item in vocab_list:
        vocab_dict[item] = i
        i+=1
    return vocab_dict




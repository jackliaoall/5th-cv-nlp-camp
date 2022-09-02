#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 21:30:16 2020

@author: krishna
"""

import numpy as np

def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])


def calc_cer(predictions,groundtruth,char_dict,print_item=False):
    rev_char_map = dict(map(reversed, char_dict.items()))
    cer_list  =[]
    select_rand = np.random.randint(0,predictions.size(0),1)
    for i in range(predictions.size(0)):
        pred_char_list = []
        org_char_list=[]
        org = groundtruth[i].detach().cpu().numpy()
        pred = predictions[i].detach().cpu().numpy()
        for k in pred:
            pred_char_list.append(rev_char_map[k])
        for k in org:
            org_char_list.append(rev_char_map[k])
        
        pred_text = ''.join(pred_char_list)
        org_text = ''.join(org_char_list)
        if print_item:
            if select_rand[0]==i:
                print('Ground truth {}'.format(org_text))
                print('Predicted text {}'.format(pred_text))
        lev_dist = levenshtein(pred_text,org_text)
        cer_list.append(lev_dist)
    return np.mean(np.asarray(lev_dist))
        
            
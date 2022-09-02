#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 13:12:43 2020

@author: krishna
"""

import os
import numpy as np
import json
import torch
import librosa


class Feature_loader(object):
    def __init__(self, json_filepath):
        super(Feature_loader, self).__init__()
        with open(json_filepath) as f:
            data = json.load(f)
        self.audio_filepath  = data['audio_filepath']
        self.chars_map = data['char_map_seq']
        self.chars = data['chars']
        self.sr = 16000
        self.feature_type = 'spectrogram'

    def load_audio(self):
        audio_data,fs  = librosa.load(self.audio_filepath,sr=self.sr)
        return audio_data    
    
    def feature_extraction(self,audio_data):
        if self.feature_type=='mfcc':
            features = librosa.feature.mfcc(audio_data, sr=self.sr,hop_length=160, win_length=400, n_mfcc=13) 
        elif self.feature_type =='mel':
            features = librosa.feature.melspectrogram(audio_data, sr=self.sr, hop_length=160, win_length=400,n_mels=40) 
        else:
            features = librosa.stft(audio_data, n_fft=512, win_length=400, hop_length=160) 
            
        return features.T
    
    def load_charmap_seq(self):
        char_map_list = [int(item)  for item in self.chars_map.split(' ')]
        return char_map_list

    def load_char_seq(self):
        chars_list = [str(item)  for item in ' '.join(self.chars.split(' _ ')).split(' ')]
        return chars_list
    
    def load_dataset(self):
        ### Audio processing
        audio_data = self.load_audio()
        features = self.feature_extraction(audio_data)#频域特征
        mag, _ = librosa.magphase(features)  # 计算幅值和相位值
        mag_T = mag.T
        spec_mag = mag_T
        # preprocessing, subtract mean, divided by time-wise var
        mu = np.mean(spec_mag, 0, keepdims=True)
        std = np.std(spec_mag, 0, keepdims=True)
        norm_spec = (spec_mag - mu) / (std + 1e-5)
        
        ### Text processing
        char_map_list = self.load_charmap_seq()
        char_list = self.load_char_seq()
        return norm_spec.T, char_map_list,char_list
    
    

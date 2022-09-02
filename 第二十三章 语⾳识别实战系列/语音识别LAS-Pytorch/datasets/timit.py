#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 18:54:36 2020

@author: krishna
"""

import os
import glob
import argparse
import json



class TIMIT(object):
    def __init__(self,config):
        super(TIMIT, self).__init__()
        self.timit_root = config.timit_dataset_root
        self.store_path = config.timit_save_root
        self.vocab_path = config.vocab_path
        
    
    def extract_words(self):
        word_list=[]
        self.train_dir = os.path.join(self.timit_root,'TRAIN')
        train_subfolders = sorted(glob.glob(self.train_dir+'/*/'))
        for sub_folder in train_subfolders:
            speaker_folders = sorted(glob.glob(sub_folder+'/*/'))
            for spk_folder in speaker_folders:
                store_folder = os.path.join(self.store_path,'TRAIN')
                if not os.path.exists(store_folder):
                    os.makedirs(store_folder)
                WAV_files = sorted(glob.glob(spk_folder+'/*.WAV'))
                for audio_filepath in WAV_files:
                    wrd_file = audio_filepath[:-4]+'.WRD'
                    read_words = [line.rstrip('\n') for line in open(wrd_file)]
                    for item in read_words:
                        word =item.split(' ')[-1].lower()
                        word_list.append(word)
        
        return word_list
    
    def create_vocab(self):
        word_list = self.extract_words()
        full_chars_list = []
        for word in word_list:
            for char in word:
                full_chars_list.append(char)
        full_chars_list.append('<s>')
        full_chars_list.append('</s>')
        full_chars_list.append('_')
        vocab = sorted(list(set(full_chars_list)))
        fid_vocab = open(self.vocab_path,'w')
        for item in vocab:
            fid_vocab.write(item+'\n')
        fid_vocab.close()
    
    def create_vocab_dict(self):
        vocab_list = [line.rstrip('\n') for line in open(self.vocab_path)]
        i=0
        vocab_dict={}
        for item in vocab_list:
            vocab_dict[item] = i
            i+=1
        return vocab_dict
    
        
    def create_char_mapping(self,word_file,vocab_dict):
        read_words = [line.rstrip('\n') for line in open(word_file)]
        char_mapped=[]
        chars_list=[]
        chars_list.append('<s>')
        char_mapped.append(vocab_dict['<s>'])
        for item in read_words:
            word =item.split(' ')[-1]
            for char in word:
                try:
                    phns = vocab_dict[char]
                except:
                    continue
                char_mapped.append(phns)
                chars_list.append(char)
            char_mapped.append(vocab_dict['_'])
            chars_list.append('_')
        char_mapped.append(vocab_dict['</s>'])
        chars_list.append('</s>')
        return char_mapped,chars_list
    
    
            
    def process_data_train(self):
        
        vocab_dict = self.create_vocab_dict()
        self.train_dir = os.path.join(self.timit_root,'TRAIN')
        train_subfolders = sorted(glob.glob(self.train_dir+'/*/'))
        for sub_folder in train_subfolders:
            speaker_folders = sorted(glob.glob(sub_folder+'/*/'))
            for spk_folder in speaker_folders:
                store_folder = os.path.join(self.store_path,'TRAIN')
                if not os.path.exists(store_folder):
                    os.makedirs(store_folder)
                WAV_files = sorted(glob.glob(spk_folder+'/*.WAV'))
                for audio_filepath in WAV_files:
                    wrd_file = audio_filepath[:-4]+'.WRD'
                    char_mapped,chars_list = self.create_char_mapping(wrd_file,vocab_dict)
                    json_write_filepath =store_folder+'/'+sub_folder.split('\\')[-2]+'_'+spk_folder.split('\\')[-2]+'_'+audio_filepath.split('\\')[-1][:-4]+'.json'
                    data_frame = {}
                    data_frame['audio_filepath'] = audio_filepath.replace('\\','/').replace('..','.')
                    data_frame['char_map_seq'] = ' '.join([str(char_item) for char_item in char_mapped])
                    data_frame['chars'] = ' '.join([str(char_item) for char_item in chars_list])
                    
                    data_frame['char_seq_len']=len(char_mapped)
                    with open(json_write_filepath, 'w') as fid:
                        json.dump(data_frame, fid,indent=4)
                        
        
    def process_data_test(self):
        vocab_dict = self.create_vocab_dict()
        self.test_dir = os.path.join(self.timit_root,'TEST')
        test_subfolders = sorted(glob.glob(self.test_dir+'/*/'))
        for sub_folder in test_subfolders:
            speaker_folders = sorted(glob.glob(sub_folder+'/*/'))
            for spk_folder in speaker_folders:
                store_folder = os.path.join(self.store_path,'TEST')
                if not os.path.exists(store_folder):
                    os.makedirs(store_folder)
                WAV_files = sorted(glob.glob(spk_folder+'/*.WAV'))
                for audio_filepath in WAV_files:
                    wrd_file = audio_filepath[:-4]+'.WRD'
                    char_mapped,chars_list = self.create_char_mapping(wrd_file,vocab_dict)
                    json_write_filepath =store_folder+'/'+sub_folder.split('\\')[-2]+'_'+spk_folder.split('\\')[-2]+'_'+audio_filepath.split('\\')[-1][:-4]+'.json'
                    data_frame = {}
                    data_frame['audio_filepath'] = audio_filepath.replace('\\','/').replace('..','.')
                    data_frame['char_map_seq'] = ' '.join([str(char_item) for char_item in char_mapped])
                    data_frame['chars'] = ' '.join([str(char_item) for char_item in chars_list])
                    
                    data_frame['char_seq_len']=len(char_mapped)
                    with open(json_write_filepath, 'w') as fid:
                        json.dump(data_frame, fid,indent=4)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Configuration for data preparation")
    parser.add_argument("--timit_dataset_root", default="../TIMIT", type=str,help='Dataset path')
    parser.add_argument("--timit_save_root", default="../TIMIT/processed_data", type=str,help='Save directory after processing')
    parser.add_argument("--vocab_path",default='../data_utils/vocab.txt',type=str, help='Filepath to write vocabulary')
    
    config = parser.parse_args()
    timit = TIMIT(config)
    timit.create_vocab() #创建语料表
    timit.process_data_train()
    timit.process_data_test()
    
    

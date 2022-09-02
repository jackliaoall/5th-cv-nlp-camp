#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 12:33:52 2020

@author: krishna

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader   
from SpeechDataGenerator import SpeechDataGenerator
import torch.nn as nn
import os
import yaml
import numpy as np
from torch import optim
import argparse
from utils.utility import create_vocab_dict
from utils.calculate_cer import calc_cer
import logging
from modules.Encoder import Encoder
from modules.Decoder import Decoder
from models.Seq2Seq import Seq2seq
from utils.utility import speech_collate
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')





def train(model,dataloader_train,epoch,optimizer,device,criterion,char_dict,LOG_FILENAME):
    total_loss=[]
    model.train()
    print('##################### Training######################')
    for i_batch, sample_batched in enumerate(dataloader_train):
        input_features = torch.stack(sample_batched[0])
        input_lengths = torch.from_numpy(np.asarray(sample_batched[1]))
        targets = torch.stack(sample_batched[2]).long()
        ground_truth_text = sample_batched[3]
        input_features = input_features.to(device)
        input_lengths = input_lengths.to(device)
        targets = targets.to(device)
     
        input_features.requires_grad = True
        optimizer.zero_grad()
        decoder_outputs,sequence_symbols = model(input_features,input_lengths,targets)
        loss = criterion(decoder_outputs,targets)
        loss.backward()
        optimizer.step()
        
        
        
        total_loss.append(loss.item())
    
    avg_cer = calc_cer(sequence_symbols,targets,char_dict,print_item=True)
    print('Training Loss {} after {} epochs'.format(np.mean(np.asarray(total_loss)),epoch))
    #print('Training CER {} after {} epochs'.format(avg_cer,epoch))
    
      
    logging.info(' total training loss %d after %d epochs', np.mean(np.asarray(total_loss)),epoch)
    
    
            
def evaluation(model,dataloader_test,epoch,criterion,device, char_dict,LOG_FILENAME):
    total_loss=[]
    model.eval()
    print('##################### Testing ######################')
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader_test):
            input_features = torch.stack(sample_batched[0])
            input_lengths = torch.from_numpy(np.asarray(sample_batched[1]))
            targets = torch.stack(sample_batched[2]).long()
            ground_truth_text = sample_batched[3]
            
            input_features = input_features.to(device)
            input_lengths = input_lengths.to(device)
            targets = targets.to(device)
        
            decoder_outputs,sequence_symbols = model(input_features,input_lengths,targets)
            loss = criterion(decoder_outputs,targets)
            
            total_loss.append(loss.item())
            if i_batch%10==0:
                avg_cer = calc_cer(sequence_symbols,targets,char_dict,print_item=True)
                print('Evaluation Loss {} after {} iteration'.format(np.mean(np.asarray(total_loss)),i_batch))
                #print('Evaluation CER {} after {} iteration'.format(avg_cer,i_batch))
            
    
    logging.info(' total testing loss %d after %d epochs', np.mean(np.asarray(total_loss)),epoch)
    


def main(config):
    
    use_cuda = config['use_gpu']
    device = torch.device("cuda" if use_cuda==1 else "cpu")
    encoder = Encoder(config['input_feat_dim'],config['enc_hidden_dim'],config['n_layers_enc'])
    decoder = Decoder(config['vocab_size'], config['max_out_len'], config['dec_hidden_dim'], config['enc_hidden_dim'],
                     config['sos_id'], config['eos_id'],config['n_layers_dec'], rnn_celltye='gru')
    model = Seq2seq(encoder,decoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    
    ### Data related
    dataset_train = SpeechDataGenerator(manifest=args.training_filepath,max_len=config['max_out_len'],pad_token=config['pad_token'])
    dataloader_train = DataLoader(dataset_train, batch_size=config['batch_size'],shuffle=True,collate_fn=speech_collate) #collate_fn：如何取样本的
    
    dataset_test = SpeechDataGenerator(manifest=args.testing_filepath,max_len=config['max_out_len'],pad_token=config['pad_token'])
    dataloader_test = DataLoader(dataset_test, batch_size=config['batch_size'] ,shuffle=True,collate_fn=speech_collate) 
    
    LOG_FILENAME='log.txt'        
    logging.info('Training Started...')
    logging.basicConfig(filename=LOG_FILENAME,level=logging.INFO)
    if not os.path.exists(args.save_modelpath):
        os.makedirs(args.save_modelpath)
        
    for epoch in range(config['num_epochs']):
        train(model,dataloader_train,epoch,optimizer,device,criterion,char_dict,LOG_FILENAME)
        evaluation(model,dataloader_test,epoch,criterion,device,char_dict,LOG_FILENAME)
        
        model_save_path = os.path.join(args.save_modelpath, 'best_check_point_'+str(epoch))
        state_dict = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch}
        torch.save(state_dict, model_save_path)
        
        


if __name__ == "__main__":
    ########## Argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-training_filepath',type=str,default='data_utils/training.txt')
    parser.add_argument('-testing_filepath',type=str, default='data_utils/testing.txt')
    parser.add_argument('-vocabulary_path',type=str, default='data_utils/vocab.txt')
    parser.add_argument('-config_file',type=str, default='data_utils/config.yaml')
    parser.add_argument('-save_modelpath',type=str, default='save_models/')
    
    args = parser.parse_args()
    
    char_dict = create_vocab_dict(args.vocabulary_path)
    with open(args.config_file) as f:
        config = yaml.safe_load(f)
    main(config)
        
          


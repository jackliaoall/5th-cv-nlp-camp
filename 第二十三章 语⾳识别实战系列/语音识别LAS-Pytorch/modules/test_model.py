#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 09:45:28 2020

@author: krishna
"""

#############
from Encoder import EncoderRNN
from Decoder import DecoderRNN
from Seq2Seq import Seq2seq
import torch


input_feat_dim = 40
enc_hidden_dim=128
n_layers_enc=2
vocab_size=2000
max_len = 100
dec_hidden_dim=256
sos_id = 0
eos_id = 0
n_layers_dec = 2
encoder = EncoderRNN(input_feat_dim,enc_hidden_dim,n_layers_enc)
decoder = DecoderRNN(vocab_size, max_len, dec_hidden_dim, enc_hidden_dim,
                 sos_id, eos_id,n_layers=1, rnn_celltye='gru')

model = Seq2seq(encoder,decoder)

## Create dummy inputs
input_features = torch.rand(10,40,300)
input_lengths = torch.randint(40,100,(10,))
target_variable = torch.randint(0,2000,(10,100))
decoder_outputs,sequence_symbols = model(input_features,input_lengths,target_variable)
import random
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from modules.attention import Attention

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device

class Decoder(nn.Module):
    def __init__(self, vocab_size, max_len, dec_hidden_size, encoder_hidden_size,
                 sos_id, eos_id,n_layers=1, rnn_celltye='gru',):
        super(Decoder, self).__init__()
        self.output_size = vocab_size
        self.vocab_size = vocab_size
        self.dec_hidden_size = dec_hidden_size
        self.encoder_output_size =encoder_hidden_size
        self.n_layers = n_layers
        self.max_length = max_len
        self.eos_id = eos_id
        self.sos_id = sos_id
        
        self.embedding = nn.Embedding(self.vocab_size, self.dec_hidden_size)
        if rnn_celltye == 'lstm':
            self.rnn = nn.LSTM(self.dec_hidden_size + self.encoder_output_size, self.dec_hidden_size, self.n_layers,
                                 batch_first=True, dropout=0.2, bidirectional=False)
        elif rnn_celltye == 'gru':
            self.rnn = nn.GRU(self.dec_hidden_size + self.encoder_output_size, self.dec_hidden_size, self.n_layers,
                            batch_first=True, dropout=0.2, bidirectional=False)
        else:
            print('Error in rnn type')
            
        self.attention = Attention(self.dec_hidden_size, self.encoder_output_size, attn_dim=self.dec_hidden_size)
        self.fc = nn.Linear(self.dec_hidden_size + self.encoder_output_size, self.output_size)
        

    def forward_step(self, input_var, hidden, encoder_outputs, context, attn_w, function):
        if self.training:
            self.rnn.flatten_parameters()
        
        if torch.cuda.is_available():
            input_var = input_var.cuda()
        embedded = self.embedding(input_var)
        #print(embedded.shape)
    
        y_all = []
        attn_w_all = []
        for i in range(embedded.size(1)):
            embedded_inputs = embedded[:, i, :]
            #print(embedded_inputs.shape)
            rnn_input = torch.cat([embedded_inputs, context], dim=1)
            #print(rnn_input.shape)
            rnn_input = rnn_input.unsqueeze(1) 
            #print(rnn_input.shape)
            output, hidden = self.rnn(rnn_input, hidden)
            #print(output.shape)
            #print(hidden.shape)
            context, attn_w = self.attention(output, encoder_outputs)
            #print(context.shape)
            #print(attn_w.shape)
            attn_w_all.append(attn_w)
            
            context = context.squeeze(1)
            #print(context.shape)
            output = output.squeeze(1) 
            #print(output.shape)
            output = torch.cat((output, context), dim=1) 
            #print(output.shape)

            pred = function(self.fc(output), dim=-1)
            #print(pred.shape)
            y_all.append(pred)

        if embedded.size(1) != 1:
            y_all = torch.stack(y_all, dim=1) 
            attn_w_all = torch.stack(attn_w_all, dim=1) 
        else:
            y_all = y_all[0].unsqueeze(1) 
            attn_w_all = attn_w_all[0] 
        
        return y_all, hidden, context, attn_w_all


    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
                    function=F.log_softmax, teacher_forcing_ratio=0):
        """
        param:inputs: Decoder inputs sequence, Shape=(B, dec_T)
        param:encoder_hidden: Encoder last hidden states, Default : None
        param:encoder_outputs: Encoder outputs, Shape=(B,enc_T,enc_D)
        """

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if teacher_forcing_ratio != 0:
            inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                                function, teacher_forcing_ratio)
        else:
            batch_size = encoder_outputs.size(0)
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            #if torch.cuda.is_available():
            #    inputs = inputs.cuda()
            max_length = self.max_length

        decoder_hidden = None
        context = encoder_outputs.new_zeros(batch_size, encoder_outputs.size(2)) # (B, D) 
        #print(context.shape) 
        attn_w = encoder_outputs.new_zeros(batch_size, encoder_outputs.size(1)) # (B, T) 
        #print(attn_w.shape)
        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output):
            decoder_outputs.append(step_output)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        if use_teacher_forcing:
            decoder_input = inputs[:, :-1]
            decoder_output, decoder_hidden, context, attn_w = self.forward_step(decoder_input, 
                                                                                decoder_hidden, 
                                                                                encoder_outputs,
                                                                                context,    
                                                                                attn_w, 
                                                                                function=function)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                decode(di, step_output)
        else:
            decoder_input = inputs[:, 0].unsqueeze(1)
            #print(decoder_input.shape)
            for di in range(max_length):
                decoder_output, decoder_hidden, context, attn_w = self.forward_step(decoder_input, 
                                                                                    decoder_hidden,
                                                                                    encoder_outputs,
                                                                                    context,
                                                                                    attn_w,
                                                                                    function=function)
                #print(decoder_output.shape)
                step_output = decoder_output.squeeze(1)
                #print(step_output.shape)
                symbols = decode(di, step_output)
                #print(symbols.shape)
                decoder_input = symbols

        return decoder_outputs,sequence_symbols
    
    
    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h
    

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        batch_size = encoder_outputs.size(0)

        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1 # minus the start of sequence symbol

        return inputs, batch_size, max_length

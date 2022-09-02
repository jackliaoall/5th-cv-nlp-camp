import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    https://arxiv.org/pdf/1506.07503.pdf
    """
    def __init__(self, dec_hidden_dim, enc_hidden_dim, attn_dim):
        super(Attention, self).__init__()
        self.dec_hidden_dim = dec_hidden_dim
        self.attn_dim = attn_dim
        self.enc_hidden_dim = enc_hidden_dim

        self.W = nn.Linear(self.dec_hidden_dim, self.attn_dim, bias=False)
        self.V = nn.Linear(self.enc_hidden_dim, self.attn_dim, bias=False)

        self.fc = nn.Linear(self.attn_dim, 1, bias=True)
        self.b = nn.Parameter(torch.rand(attn_dim))

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)



    def forward(self,Si_1,Hj):
        score =  self.fc(self.tanh(
         self.W(Si_1) + self.V(Hj)  + self.b
        )).squeeze(dim=-1)
        #print(score.shape)
        attn_weight = self.softmax(score) 
        #print(attn_weight.shape)
        context = torch.bmm(attn_weight.unsqueeze(dim=1), Hj)
        #print(context.shape)

        return context, attn_weight
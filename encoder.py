import torch 
from torch import nn
import math
import torch.nn.functiona as F

def scaled_dot_product(q,k,v,mask=None):
    d_k=q.size()[-1]
    scaled=torch.matmul(q,k.transpose(-1,-2))/math.sqrt(d_k)
    print(f"sclaed.size: {scaled.size()}")
    if mask is not None:
        scaled+=mask
    attention=F.softmax(sclaed,dim=1)
    values= torch.matmul(attention,v)
    return values,attention

class MultiHeadedAttention(nn.Module):
    




class Encoder(nn.Module):
    def __init__(self,d_model,ffn_hidden,num_heads,drop_prob,num_layers):
        super.__init__()
        self.layers=nn.Sequential(*[EncoderLayer(d_model,ffn_hidden,num_heads,drop_prob)
                                    for _ in range(num_layers)])
    def forward(self,x):
        x=self.layers(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,num_heads,drop_prob):
        super(EncoderLayer,self).__init__()
        self.attention=MultiHeadAttention(d_model=d_model,num_heads=num_heads)
        self.norm1= LayerNormalization()
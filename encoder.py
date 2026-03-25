import torch 
from torch import nn
import math
import torch.nn.functional as F

def scaled_dot_product(q,k,v,mask=None):
    d_k=q.size()[-1]
    scaled=torch.matmul(q,k.transpose(-1,-2))/math.sqrt(d_k)
    print(f"sclaed.size: {scaled.size()}")
    if mask is not None:
        scaled+=mask
    attention=F.softmax(scaled,dim=1)
    values= torch.matmul(attention,v)
    return values,attention

class MultiHeadedAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.qkv_layer=nn.Linear(d_model,3*d_model)
        self.linear_layer=nn.Linear(d_model,d_model)
    def forward(self,x,mask=None):
        batch_size,max_seq_length,d_model=x.size()
        qkv=self.qkv_layer(x)
        qkv=qkv.reshape(batch_size,max_seq_length,self.num_heads,3*self.head_dim)
        qkv=qkv.permute(0,2,1,3)
        q,k,v=qkv.chunk(3,dim=-1)
        values,attention= scaled_dot_product(q,k,v,mask)
        values=values.reshape(batch_size,max_seq_length,self.num_heads)
        out=self.linear_layer(values)
        return out



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
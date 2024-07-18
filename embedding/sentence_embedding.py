import torch
import math
from torch import nn
class embedding(nn.Module):
    def __init__(self,input,hidden,output):
        super(embedding,self).__init__()
        self.embed_len=output
        self.Layer1=nn.Linear(input,hidden)
        self.Layer2=nn.Linear(hidden,output)

    def forward(self, x):
        # 保证 x 的形状和 input 匹配
        ret1 = torch.sigmoid(self.Layer1(x))
        ret2 = self.Layer2(ret1)
        
        # 增加位置机制
        d_model = ret2.shape[2]
        pe = torch.zeros_like(ret2, dtype=torch.float32)

        position = torch.arange(0, x.shape[1], dtype=torch.float32, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=x.device) * -(math.log(10000.0) / d_model))

        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term[:d_model//2])

        ret2 = ret2 + pe
        
        return ret2
    


        
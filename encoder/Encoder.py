from torch import nn
from self_attention.attention import SelfAttention
from self_attention.give_mask import Mask

class Encoder_model(nn.Module):
    def __init__(self,batch_size,embed_size,head_num):
        super(Encoder_model,self).__init__()
        self.attention_block=SelfAttention(batch_size,embed_size,head_num)

        #直接写死，中间层的宽带是输入层的4倍  来自论文
        hidden_size=embed_size*4
        self.Linear1=nn.Linear(embed_size,hidden_size)
        self.Linear2=nn.Linear(hidden_size,embed_size)
        self.relu=nn.ReLU()

    def forward(self,x,mask):
        ret_attention=self.attention_block(x,mask=mask,KV=None)
        ret_attention+=x   #残差层

        ret_linear1=self.relu(self.Linear1(ret_attention))
        ret_linear2=self.Linear2(ret_linear1)
        ret=ret_linear2+ret_attention   #残差层

        return ret
    
class Encoder_stack(nn.Module):
    def __init__(self, stack_num, batch_size,embed_size, head_num):
        super(Encoder_stack, self).__init__()
        self.model_stack = nn.ModuleList([Encoder_model(batch_size,embed_size, head_num) for _ in range(stack_num)])

    def forward(self, x,mask):
        for encoder in self.model_stack:
            x = encoder(x,mask)
        return x
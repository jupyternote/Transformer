import torch
from torch import nn
from self_attention.attention import SelfAttention
from self_attention.give_mask import Mask

class Decoder_model(nn.Module):
    def __init__(self, batch_size,embed_size,head_num):
        super(Decoder_model, self).__init__()
        
        self.Mask_attention = SelfAttention(batch_size, embed_size, head_num)
        self.attention = SelfAttention(batch_size, embed_size, head_num)

        hidden_size = embed_size * 4
        self.Linear1 = nn.Linear(embed_size, hidden_size)
        self.Linear2 = nn.Linear(hidden_size, embed_size)

        self.relu = nn.ReLU()

    def forward(self, x, KV, mask1, mask2):
        mask_attention = self.Mask_attention(x, KV=None,mask=mask1) + x
        attention = self.attention(mask_attention, KV, mask2) + mask_attention
        ret = self.Linear2(self.relu(self.Linear1(attention))) + attention
        return ret

class Decoder_stack(nn.Module):
    def __init__(self, stack_num, batch_size,embed_size, head_num):
        super(Decoder_stack, self).__init__()
        self.model_stack = nn.ModuleList([Decoder_model(batch_size,embed_size, head_num) for _ in range(stack_num)])

    def forward(self, x, KV, mask1, mask2):
        for decoder in self.model_stack:
            x = decoder(x, KV, mask1, mask2)
        return x
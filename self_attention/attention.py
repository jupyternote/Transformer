import torch
from torch import nn
from torch.nn.functional import softmax

class SelfAttention(nn.Module):
    def __init__(self, batch_size, embed_size, head_num):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.head_num = head_num
        self.batch_size = batch_size

        # 确保embed_size和head_num之间的关系
        if embed_size % head_num != 0:
            raise ValueError("embed_size must be divisible by head_num")

        self.head_dim = embed_size // head_num

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x, KV=None, mask=None):
        seq_length = x.shape[1]

        # 计算QKV
        Q = self.query(x).reshape(self.batch_size, seq_length, self.head_num, self.head_dim).transpose(1, 2)

        if KV is not None:
            K = self.key(KV).reshape(self.batch_size, KV.shape[1], self.head_num, self.head_dim).transpose(1, 2)
            V = self.value(KV).reshape(self.batch_size, KV.shape[1], self.head_num, self.head_dim).transpose(1, 2)
        else:
            K = self.key(x).reshape(self.batch_size, seq_length, self.head_num, self.head_dim).transpose(1, 2)
            V = self.value(x).reshape(self.batch_size, seq_length, self.head_num, self.head_dim).transpose(1, 2)

        # 计算score
        score = torch.einsum('bnqd,bnkd->bnqk', [Q, K]) / (self.head_dim ** 0.5)

        if mask is not None:
            score = score.masked_fill(mask == False, float("-1e20"))

        weights = softmax(score, dim=-1)

        output = torch.einsum("bnql,bnld->bnqd", [weights, V]).transpose(1, 2).contiguous().view(self.batch_size, seq_length, self.embed_size)

        return self.fc_out(output)

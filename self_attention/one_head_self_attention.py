import torch
from torch import nn
class SelfAttention(nn.Module):
    def __init__(self,embed_size,output_size=-1):
        super(SelfAttention,self)
        
        # 之所以输出维度选择embed_size，是为了多层堆叠过程方便，不用指定每层大小
        if output_size==-1:
            output_size=embed_size

        self.embed_size=embed_size
        self.output_size=output_size

        # nn.Linear可以处理任意维度的张量，针对最里面的维度进行全连接处理     
        self.query=nn.Linear(embed_size,output_size)
        self.key=nn.Linear(embed_size,output_size)
        self.value=nn.Linear(embed_size,output_size)

    def forward(self,x):
        """
            原理剖析：
            score计算了任何两个单词之间的相似度
            weights是对相似度打分进行了归一化
            output对相似度进行加权
        """
        #计算Q K V
        Q=self.query(x)
        K=self.Key(x)
        V=self.value(x)

        #计算自注意力分数
        score=torch.matmul(Q,K.transpose(-1,-2))/(self.embed_size**0.5)
        #通过在单词上softmax得到权重，每个位置都是两两相似度得分
        weights=nn.functional.softmax(score,dim=-1)
        #和value获得得分，相当于加权
        output=torch.matmul(weights,V)
        return output

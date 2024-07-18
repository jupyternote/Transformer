import torch
from torch import nn
class SelfAttention(nn.Module):
    def __init__(self,embed_size,output_size=-1):
        super(SelfAttention,self)
        
        # ֮�������ά��ѡ��embed_size����Ϊ�˶��ѵ����̷��㣬����ָ��ÿ���С
        if output_size==-1:
            output_size=embed_size

        self.embed_size=embed_size
        self.output_size=output_size

        # nn.Linear���Դ�������ά�ȵ�����������������ά�Ƚ���ȫ���Ӵ���     
        self.query=nn.Linear(embed_size,output_size)
        self.key=nn.Linear(embed_size,output_size)
        self.value=nn.Linear(embed_size,output_size)

    def forward(self,x):
        """
            ԭ��������
            score�������κ���������֮������ƶ�
            weights�Ƕ����ƶȴ�ֽ����˹�һ��
            output�����ƶȽ��м�Ȩ
        """
        #����Q K V
        Q=self.query(x)
        K=self.Key(x)
        V=self.value(x)

        #������ע��������
        score=torch.matmul(Q,K.transpose(-1,-2))/(self.embed_size**0.5)
        #ͨ���ڵ�����softmax�õ�Ȩ�أ�ÿ��λ�ö����������ƶȵ÷�
        weights=nn.functional.softmax(score,dim=-1)
        #��value��õ÷֣��൱�ڼ�Ȩ
        output=torch.matmul(weights,V)
        return output

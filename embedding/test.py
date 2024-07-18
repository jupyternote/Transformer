import torch
import sentence_embedding
batch_size=20
seq_length=100
input_size=256
hidden_size=1024
embedding_size=512
input=torch.rand(batch_size,seq_length,input_size)
model=sentence_embedding.embedding(input_size,hidden_size,embedding_size)
out=model(input)
print(out.size())    #(batch_size,len_sequence,embedding_size)

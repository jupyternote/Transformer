from torch import nn
from self_attention.give_mask import padding_mask, Mask
from embedding.sentence_embedding import embedding
from encoder.Encoder import Encoder_stack
from decoder.Decoder import Decoder_stack

class Transformer(nn.Module):
    def __init__(self, input_len=256, hidden_len=512, embedding_size=512, stack_num=6, head_num=8, batch_size=10, output_hidden=2048, output_out=1024):
        super(Transformer, self).__init__()

        # 定义常量
        self.input_len = input_len
        self.hidden_len = hidden_len
        self.embedding_size = embedding_size
        self.batch_size = batch_size

        # 嵌入层
        self.InputEmbedding = embedding(self.input_len, self.hidden_len, self.embedding_size)
        self.OutputEmbedding = embedding(self.input_len, self.hidden_len, self.embedding_size)

        # 编码器和解码器
        self.encoder = Encoder_stack(stack_num=stack_num, embed_size=self.embedding_size, head_num=head_num, batch_size=batch_size)
        self.decoder = Decoder_stack(stack_num=stack_num, embed_size=self.embedding_size, head_num=head_num, batch_size=batch_size)

        # 输出层
        self.output_hidden = output_hidden
        self.output_out = output_out
        self.output_linear1 = nn.Linear(self.embedding_size, self.output_hidden)
        self.output_linear2 = nn.Linear(self.output_hidden, self.output_out)

    def forward(self, input, output, enc_mask, dec_mask):

        # 输入嵌入
        input_embed = self.InputEmbedding(input)
        output_embed = self.OutputEmbedding(output)

        # 编码器输出
        enc_output = self.encoder(input_embed, enc_mask)

        # 解码器输出
        dec_output = self.decoder(output_embed, enc_output, dec_mask,enc_mask)

        # 全连接层
        linear_out = self.output_linear2(nn.ReLU()(self.output_linear1(dec_output)))

        return linear_out

        # Softmax 输出
        #return nn.functional.softmax(linear_out, dim=-1)




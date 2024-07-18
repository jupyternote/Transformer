import torch
"""
    mask有两种 padding mask和 look_ahead mask
    padding mask用来把句子填充到相同的长度
    look_ahead mask防止decoder看到后面的东西
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def padding_mask(sentence, padding_value=-2):
    mask = (sentence != padding_value)
    mask = mask[:, :, 0]
    mask = mask.unsqueeze(1).unsqueeze(2)
    return mask

def Mask(sentence, padding_value=-2):
    # 假设 sentence 的形状为 (batch_size, seq_length, embed_dim)
    # 创建一个布尔掩码，掩盖填充值的位置
    mask = (sentence != padding_value)
    
    # 去掉嵌入维度，只保留 batch_size 和 seq_length
    mask = mask[:, :, 0]
    
    # 创建下三角掩码，确保只关注当前时间步及之前的信息
    seq_length = mask.shape[1]
    subsequent_mask = torch.tril(torch.ones((seq_length, seq_length))).bool()
    subsequent_mask=subsequent_mask.to(device)
    # 扩展掩码的形状以适应多头自注意力
    mask = mask.unsqueeze(1).unsqueeze(2).to(device)
    combined_mask = mask & subsequent_mask.unsqueeze(0).unsqueeze(0)
    
    return combined_mask


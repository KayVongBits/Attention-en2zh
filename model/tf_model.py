import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config

DEVICE = config.DEVICE
class EmbeddingLayer(nn.Module):
    '''Embedding layer for the model.
    Args:
        vocab_size: The size of the vocabulary.
        embedding_dim: The dimension of the embedding vectors.
    '''
    def __init__(self, vocab_size, embedding_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dim = embedding_dim

    def forward(self, x):

        return self.embedding(x) * math.sqrt(self.dim)  # 乘以sqrt(d_model)是为了在训练初期保持嵌入向量的尺度稳定，防止过大或过小的值导致训练不稳定。
    
class PositionalEncoding(nn.Module):
    '''Positional encoding layer for the model.
    Args:
        d_model: The dimension of the model.
        max_len: The maximum length of the input sequences.
    '''
    def __init__(self, d_model, max_len=5000,dropout=0.1):
        self.dropout = nn.Dropout(p=dropout)

        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model,device=DEVICE)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)            # unsqueeze(0)将pe的形状从(max_len, d_model)变为(1, max_len, d_model)，以便在后续的计算中能够正确地与输入的嵌入向量进行广播操作。
        self.register_buffer('pe', pe)  # 将pe矩阵以持久的buffer状态存下(不会作为要训练的参数)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
def attention(query, key, value, mask=None, dropout=None):
    '''Compute 'Scaled Dot Product Attention'
    Args:
        query: Queries of shape (batch_size, h, seq_len, d_k)
        key: Keys of shape (batch_size, h, seq_len, d_k)
        value: Values of shape (batch_size, h, seq_len, d_v)
        mask: Optional mask of shape (batch_size, 1, seq_len) or (batch_size, seq_len, seq_len)
        dropout: Optional dropout layer
    Returns:
        output: Attention output of shape (batch_size, h, seq_len, d_v)
        attn: Attention weights of shape (batch_size, h, seq_len, seq_len)
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # 计算注意力分数，并进行缩放
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # 将mask位置的分数设置为一个非常小的值，确保这些位置在softmax后几乎为0
    attn = F.softmax(scores, dim=-1)  # 对注意力分数进行softmax，得到注意力权重
    if dropout is not None:
        attn = dropout(attn)  # 可选的dropout层应用于注意力权重
    output = torch.matmul(attn, value)  # 将注意力权重与value矩阵相乘，得到最终的输出
    return output, attn
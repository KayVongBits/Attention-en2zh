import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config
import copy

DEVICE = config.device

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
        scores = scores.masked_fill(mask == 0, -1e9)  # 使用masked_fill将mask中为0的位置的分数设置为一个非常大的负数（-1e9），这样在softmax计算时，这些位置的权重将接近于0，从而有效地忽略这些位置的影响。
    
    attn = F.softmax(scores, dim=-1)  # 对注意力分数进行softmax，得到注意力权重
    
    if dropout is not None:
        attn = dropout(attn)  # 可选的dropout层应用于注意力权重
    
    output = torch.matmul(attn, value)  # 将注意力权重与value矩阵相乘，得到最终的输出
    
    return output, attn

class MultiHeadAttention(nn.Module):
    '''Multi-head attention layer for the model.
    Args:
        h: The number of attention heads.
        d_model: The dimension of the model.
        dropout: The dropout rate.
    '''
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0  # 确保d_model可以被h整除，以便每个头的维度是整数

        self.d_k = d_model // h  # 每个头的维度
        self.h = h  # 注意力头的数量

        self.linear_q = nn.Linear(d_model, d_model)  # 用于生成查询向量的线性层
        self.linear_k = nn.Linear(d_model, d_model)  # 用于生成键向量的线性层
        self.linear_v = nn.Linear(d_model, d_model)  # 用于生成值向量的线性层

        self.dropout = nn.Dropout(p=dropout)  # Dropout层
        self.out_proj = nn.Linear(d_model, d_model)  # 输出线性层，将多头注意力的输出映射回d_model维度

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)  # 将mask的形状从(batch_size, seq_len)变为(batch_size, 1, seq_len)，以便在计算注意力分数时能够正确地广播到所有的注意力头和序列位置。

        batch_size = query.size(0)# 获取批次大小

        # 线性变换并分割成多个头
        query = self.linear_q(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_len, d_k)
        key = self.linear_k(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)      # (batch_size, h, seq_len, d_k)
        value = self.linear_v(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_len, d_k)

        # 计算注意力
        x, attn = attention(query, key, value, mask=mask, dropout=self.dropout)  # x: (batch_size, h, seq_len, d_k)

        # 将多头的输出连接起来，并通过输出线性层
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)  # (batch_size, seq_len, d_model)
        return self.out_proj(x)  # 输出最终的多头注意力结果
       

class LayerNorm(nn.Module):
    '''Layer normalization layer for the model.
    Args:
        features: The number of features in the input.
        eps: A small value to prevent division by zero.
    ''' 
    def __init__(self, features, eps=1e-6):

        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * out + self.beta
    
class PositionwiseFeedForward(nn.Module):
    '''Position-wise feedforward layer for the model.
    Args:
        d_model: The dimension of the model.
        d_ff: The dimension of the feedforward layer.
        dropout: The dropout rate.
    '''
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # 第一个线性层，将输入从d_model维度映射到d_ff维度
        self.linear2 = nn.Linear(d_ff, d_model)  # 第二个线性层，将输出从d_ff维度映射回d_model维度
        self.dropout = nn.Dropout(p=dropout)  # Dropout层

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))  # 前馈网络的前向传播，包含ReLU激活和Dropout

class SublayerConnection(nn.Module):
    '''A residual connection followed by a layer norm.
    Args:
        size: The dimension of the model.
        dropout: The dropout rate.
    '''
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)  # Layer normalization层
        self.dropout = nn.Dropout(p=dropout)  # Dropout层

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))  # 残差连接，将输入x与子层的输出相加，并应用Dropout

def clones(module, N):
    '''Produce N identical layers.
    Args:
        module: The module to be cloned.
        N: The number of clones to produce.
    Returns:
        A ModuleList containing N clones of the input module.
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])# 使用copy.deepcopy确保每个克隆的模块都是独立的实例，具有自己的参数和状态。
    #不要使用[module] * N，因为这会创建N个对同一模块的引用，而不是N个独立的模块实例。

class EncoderLayer(nn.Module):
    '''Encoder layer for the model.
    Args:
        size: The dimension of the model.
        self_attn: The self-attention module.
        feed_forward: The feedforward module.
        dropout: The dropout rate.
    '''
    def __init__(self, size, self_attn, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn  # 自注意力模块
        self.feed_forward = feed_forward  # 前馈网络模块
        self.sublayer = clones(SublayerConnection(size, dropout), 2)  # 两个子层连接，分别用于自注意力和前馈网络
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))  # 第一个子层连接，应用自注意力
        return self.sublayer[1](x, self.feed_forward)  # 第二个子层连接，应用前馈网络
    
class Encoder(nn.Module):
    '''Core encoder is a stack of N layers.
    Args:
        layer: An instance of the EncoderLayer class.
        N: The number of layers to stack.
    '''
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)  # 克隆N个编码器层
        self.norm = LayerNorm(layer.size)  # 最后的层归一化

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)  # 依次通过每个编码器层
        return self.norm(x)  # 最后进行层归一化
    
class DecoderLayer(nn.Module):
    '''Decoder layer for the model.
    Args:
        size: The dimension of the model.
        self_attn: The self-attention module.
        src_attn: The source attention module.
        feed_forward: The feedforward module.
        dropout: The dropout rate.
    '''
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn  # 自注意力模块
        self.src_attn = src_attn  # 源注意力模块，用于解码器对编码器输出的注意力计算
        self.feed_forward = feed_forward  # 前馈网络模块
        self.sublayer = clones(SublayerConnection(size, dropout), 3)  # 三个子层连接，分别用于自注意力、源注意力和前馈网络
        self.size = size

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory  # 编码器的输出作为memory输入
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))  # 第一个子层连接，应用自注意力
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))  # 第二个子层连接，应用源注意力
        return self.sublayer[2](x, self.feed_forward)  # 第三个子层连接，应用前馈网络
    
class Decoder(nn.Module):
    '''Generic N layer decoder with masking.
    Args:
        layer: An instance of the DecoderLayer class.
        N: The number of layers to stack.
    '''
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)  # 克隆N个解码器层
        self.norm = LayerNorm(layer.size)  # 最后的层归一化

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)  # 依次通过每个解码器层
        return self.norm(x)  # 最后进行层归一化
    
class Generator(nn.Module):
    '''Define standard linear + softmax generation step.
    Args:
        d_model: The dimension of the model.
        vocab_size: The size of the target vocabulary.
    '''
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)  # 线性层，将模型输出映射到词汇表大小的维度

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)  # 对线性层的输出进行log softmax，得到每个词的对数概率分布
    

class Transformer(nn.Module):
    '''The Transformer model.
    Args:
        encoder: The encoder module.
        decoder: The decoder module.
        src_embed: The source embedding module.
        tgt_embed: The target embedding module.
        generator: The generator module.
    '''
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder  # 编码器模块
        self.decoder = decoder  # 解码器模块
        self.src_embed = src_embed  # 源语言嵌入模块
        self.tgt_embed = tgt_embed  # 目标语言嵌入模块
        self.generator = generator  # 生成模块

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encoder(self.src_embed(src), src_mask)  # 编码器处理源输入，得到memory
        output = self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)  # 解码器处理目标输入和memory，得到输出
        return self.generator(output)  # 通过生成模块得到最终的输出概率分布
    
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    '''Helper: Construct a model from hyperparameters.
    Args:
        src_vocab: The size of the source vocabulary.
        tgt_vocab: The size of the target vocabulary.
        N: The number of layers in the encoder and decoder stacks.
        d_model: The dimension of the model.
        d_ff: The dimension of the feedforward layer.
        h: The number of attention heads.
        dropout: The dropout rate.
    Returns:
        An instance of the Transformer model.
    '''
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model, dropout)  # 创建多头注意力模块
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)  # 创建前馈网络模块
    position = PositionalEncoding(d_model, dropout=dropout)  # 创建位置编码模块
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),  # 创建编码器，包含N个编码器层
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),  # 创建解码器，包含N个解码器层
        nn.Sequential(EmbeddingLayer(src_vocab, d_model), c(position)),  # 创建源语言嵌入模块，包含嵌入层和位置编码
        nn.Sequential(EmbeddingLayer(tgt_vocab, d_model), c(position)),  # 创建目标语言嵌入模块，包含嵌入层和位置编码
        Generator(d_model, tgt_vocab))  # 创建生成模块

    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)  # 使用Xavier均匀分布初始化模型参数，以帮助模型更快地收敛

    return model.to(DEVICE)
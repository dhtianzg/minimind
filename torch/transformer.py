import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class MultiHeadAtention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAtention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads


        # 定义线性变换层（无需偏置）
        self.W_q = nn.Linear(d_model, d_model) # 查询变换
        self.W_k = nn.Linear(d_model, d_model) # 键变换
        self.W_v = nn.Linear(d_model, d_model) # 值变换
        self.W_o = nn.Linear(d_model, d_model) # 输出变换

    def split_heads(self, x):
        """"
        将输入张量 x 拆分为多个头
        输入: x 形状为 (batch_size, seq_len, d_model)
        输出: 形状为 (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()
        # 先将最后一维拆分为 (num_heads, d_k)，再调整维度顺序
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """
        将多个头的输出合并为一个张量
        输入: x 形状为 (batch_size, num_heads, seq_len, d_k)
        输出: 形状为 (batch_size, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.size()
        # 调整维度顺序，再将最后两维合并为 d_model
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        计算缩放点积注意力
        输入:
            Q: 查询张量，形状为 (batch_size, num_heads, seq_len_q, d_k)
            K: 键张量，形状为 (batch_size, num_heads, seq_len_k, d_k)
            V: 值张量，形状为 (batch_size, num_heads, seq_len_v, d_k)
            mask: 可选的掩码张量，形状为 (batch_size, 1, seq_len_q, seq_len_k)
        输出:
            输出张量，形状为 (batch_size, num_heads, seq_len_q, d_k)
        """
        # 计算注意力分数（Q和K的点积）
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
       
        # 应用掩码（如填充掩码或未来信息掩码）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
       
        # 计算注意力权重（softmax归一化）
        attn_probs = torch.softmax(attn_scores, dim=-1)
       
        # 对值向量加权求和
        output = torch.matmul(attn_probs, V)
        return output

    def forward(self, Q, K, V, mask=None):
            """
            前向传播
            输入形状: Q/K/V: (batch_size, seq_length, d_model)
            输出形状: (batch_size, seq_length, d_model)
            """
            # 线性变换并分割多头
            Q = self.split_heads(self.W_q(Q)) # (batch, heads, seq_len, d_k)
            K = self.split_heads(self.W_k(K))
            V = self.split_heads(self.W_v(V))
        
            # 计算注意力
            attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
            # 合并多头并输出变换
            output = self.W_o(self.combine_heads(attn_output))
            return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # 第一层全连接
        self.fc2 = nn.Linear(d_ff, d_model)  # 第二层全连接
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, x):
        # 前馈网络的计算
        return self.fc2(self.relu(self.fc1(x)))
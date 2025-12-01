import math

import torch
from torch import nn

from config.arsenal_model_config import ArsenalConfig


class ArsenalAttention(nn.Module):
    """多头注意力实现来自论文'Attention Is All You Need'
    论文链接:https://arxiv.org/abs/1706.03762"""

    def __init__(self, config: ArsenalConfig):
        super(ArsenalAttention, self).__init__()
        self.config = config
        self.head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("注意力头必须是隐藏层大小的整数倍")
        # 正则化采用丢弃的方式
        self.dropout = nn.Dropout(config.attention_dropout)
        # 初始化Q,K,V
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim,
                                bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim,
                                bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim,
                                bias=config.attention_bias)
        # 初始化输出的线性层
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size,
                                bias=config.attention_bias)
        # 注册注意力掩码
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(config.max_position_embedding, config.max_position_embedding), diagonal=1)
        )

    def forward(self, hidden_states: torch.Tensor):
        bsz, seq_len, _ = hidden_states.shape

        # Query计算
        # Shape变化:(bsz,seq_len,hidden_size)->(bsz,seq_len,config.num_attention_heads * self.head_dim)
        q_states = self.q_proj(hidden_states)
        # 改变Q的Shape为:(bsz,seq_len,hidden_size)->(bsz,seq_len,num_attention_heads,head_dim)
        q_states = q_states.view(bsz, seq_len, self.config.num_attention_heads, self.head_dim)
        # 将Q转置为:(bsz,num_attention_heads,seq_len,head_dim)
        q_states = q_states.transpose(1, 2)

        # Key计算
        # Shape变化:(bsz,seq_len,hidden_size)->(bsz,seq_len,config.num_attention_heads * self.head_dim)
        k_states = self.k_proj(hidden_states)
        # 改变Shape为:(bsz,seq_len,hidden_size)->(bsz,seq_len,num_attention_heads,head_dim)
        k_states = k_states.view(bsz, seq_len, self.config.num_attention_heads, self.head_dim)
        # 转置为:(bsz,num_attention_heads,seq_len,head_dim)
        k_states = k_states.transpose(1, 2)

        # Value计算
        # Shape变化:(bsz,seq_len,hidden_size)->(bsz,seq_len,config.num_attention_heads * self.head_dim)
        v_states = self.v_proj(hidden_states)
        # 改变Shape为:(bsz,seq_len,hidden_size)->(bsz,seq_len,num_attention_heads,head_dim)
        v_states = v_states.view(bsz, seq_len, self.config.num_attention_heads, self.head_dim)
        # 转置为:(bsz,num_attention_heads,seq_len,head_dim)
        v_states = v_states.transpose(1, 2)

        # 计算注意力权重
        # (bsz,num_attention_heads,seq_len,head_dim) * (bsz,num_attention_heads,head_dim,seq_len) = (bsz,num_attention_heads,seq_len,seq_len)
        attn_weights = torch.matmul(q_states, k_states.transpose(2, 3))

        # 应用注意力掩码
        mask_bool = self.mask.bool()[:seq_len, :seq_len]
        attn_weights.masked_fill_(mask_bool, -math.inf)

        # 计算缩放点积注意力
        attn_weights = attn_weights / math.sqrt(self.head_dim)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # 正则化
        attn_weights = self.dropout(attn_weights)

        # 输出结果
        # (bsz,num_attention_heads,seq_len,seq_len) * (bsz,num_attention_heads,seq_len,head_dim) = (bsz,num_attention_heads,seq_len,head_dim)
        attn_outputs = torch.matmul(attn_weights, v_states)
        # (bsz,num_attention_heads,seq_len,head_dim) -> (bsz,seq_len,num_attention_heads,head_dim)
        attn_outputs = attn_outputs.transpose(1, 2).contiguous()
        # (bsz,seq_len,num_attention_heads,head_dim) -> (bsz,seq_len,hidden_size)
        attn_outputs = attn_outputs.reshape(bsz, seq_len, self.hidden_size)
        # (bsz,seq_len,hidden_size)->(bsz,seq_len,hidden_size)
        attn_outputs = self.o_proj(attn_outputs)
        return attn_outputs, attn_weights

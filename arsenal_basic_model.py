from typing import Optional, Tuple

import torch
from torch import nn
from transformers import AutoTokenizer

from config.arsenal_model_config import ArsenalConfig

tokenizer = AutoTokenizer.from_pretrained('tokenization', trust_remote_code=True, pad_token='<|endoftext|>')


class ArsenalAttention(nn.Module):
    """多头注意力实现来自论文'Attention Is All You Need'
    论文链接:https://arxiv.org/abs/1706.03762"""

    def __init__(self, config: ArsenalConfig, layer_idx):
        super(ArsenalAttention, self).__init__()
        self.config = config
        self.layer_idx = layer_idx
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

    def forward(self, hidden_states: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor]):
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
        # 将K转置为:(bsz,num_attention_heads,seq_len,head_dim)
        k_states = k_states.transpose(1, 2)

        # 计算q,k旋转位置编码
        cos, sin = position_embeddings
        q_states, k_states = apply_rotary_pos_emb(q_states, k_states, cos, sin)

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
        attn_weights.masked_fill_(mask_bool, -torch.inf)

        # 计算缩放点积注意力
        attn_weights = attn_weights / self.head_dim ** 0.5
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # 正则化
        attn_weights = self.dropout(attn_weights)

        # 输出结果
        # (bsz,num_attention_heads,seq_len,seq_len) * (bsz,num_attention_heads,seq_len,head_dim) = (bsz,num_attention_heads,seq_len,head_dim)
        attn_outputs = torch.matmul(attn_weights, v_states)
        # (bsz,num_attention_heads,seq_len,head_dim) -> (bsz,seq_len,num_attention_heads,head_dim)
        attn_outputs = attn_outputs.transpose(1, 2).contiguous()
        # (bsz,seq_len,num_attention_heads,head_dim) -> (bsz,seq_len,hidden_size)
        attn_outputs = attn_outputs.reshape(bsz, seq_len, self.config.hidden_size)
        # (bsz,seq_len,hidden_size)->(bsz,seq_len,hidden_size)
        attn_outputs = self.o_proj(attn_outputs)
        return attn_outputs, attn_weights


def compute_rope_freq(config: ArsenalConfig, position_ids):
    """
    计算旋转频率
    position_ids(1,max_position_embedding)
    """
    # 获取基础频率
    rope_base = config.rope_base
    # 获取嵌入维度
    dim = config.head_dim
    # 初始化注意力因子为1.0
    attn_factor = 1.0
    # 两两分组,每组元素应该旋转的频率size=(dim/2)
    inv_freq = 1.0 / (rope_base ** (
            torch.arange(0, dim, 2, dtype=torch.int64) / dim))
    # (dim/2)->(1,dim/2,1)->(1,dim/2,1)
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    # (1,max_position_embedding)->(1,1,max_position_embedding)
    position_ids_expanded = position_ids[:, None, :].float()
    # 计算对应的cos和sin的值(1,dim/2,1))@(1,1,max_position_embedding)=>(1,dim/2,max_position_embedding)=>(1,max_position_embedding,dim/2)
    freq_s = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    # (1,max_position_embedding,dim/2)=>(1,max_position_embedding,dim)
    emb = torch.cat((freq_s, freq_s), dim=-1)
    cos = emb.cos() * attn_factor
    sin = emb.sin() * attn_factor
    return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    应用旋转位置编码
    """
    # (1,max_position_embedding,dim)->(1,1,max_position_embedding,dim)
    cos = cos.unsqueeze(unsqueeze_dim)
    # (1,max_position_embedding,dim)->(1,1,max_position_embedding,dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # q->(bsz,seq_len,num_attention_heads,head_dim) * (1,1,max_position_embedding,dim) -> (bsz,num_attention_heads,max_position_embedding,dim)
    # k->(bsz,seq_len,num_attention_heads,head_dim) * (1,1,max_position_embedding,dim) -> (bsz,num_attention_heads,max_position_embedding,dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    # (bsz,num_attention_heads,seq_len,head_dim) -> (bsz,num_attention_heads,seq_len,head_dim/2)
    # 取前部分
    x1 = x[..., : x.shape[-1] // 2]
    # 取后部分
    # (bsz,num_attention_heads,seq_len,head_dim) -> (bsz,num_attention_heads,seq_len,head_dim/2)
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class ArsenalLayerNorm(nn.Module):
    """使用层归一化
        公式为:(样本-均值/(方差+很小的数(默认1e-5)防止除零)的平方根)*缩放+偏移
    """

    def __init__(self, hidden_size, eps=1e-5):
        super(ArsenalLayerNorm, self).__init__()
        self.eps = eps
        # 缩放和偏移
        # scale和shift是可学习的参数让模型能够适应不同的数据分布
        # 缩放被初始化为1,由公式可知应该被初始化为1
        self.scale = nn.Parameter(torch.ones(hidden_size))
        # 偏移被初始化为0,由公式可知应该被初始化为0
        self.shift = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        # 计算均值
        mean = hidden_states.mean(dim=-1, keepdim=True)
        # 计算方差
        variance = hidden_states.var(dim=-1, keepdim=True, unbiased=False)
        # 套入层归一化公式
        hidden_states_normalized = (hidden_states - mean) / torch.sqrt(variance + self.eps)
        # 返回结果
        return hidden_states_normalized * self.scale + self.shift


class FeedForward(nn.Module):
    """transformer架构中前馈神经网络处理"""

    def __init__(self, config: ArsenalConfig):
        super(FeedForward, self).__init__()
        self.hidden_size = config.hidden_size
        self.layers = nn.Sequential(
            nn.Linear(self.hidden_size, config.intermediate_size),
            # 使用GELU激活函数
            nn.GELU(),
            nn.Linear(config.intermediate_size, self.hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor):
        return self.layers(hidden_states)


class ArsenalDecoderLayer(nn.Module):
    """
    transformer架构中Decoder层的实现
    """

    def __init__(self, config: ArsenalConfig, layer_idx):
        super(ArsenalDecoderLayer, self).__init__()
        self.hidden_size = config.hidden_size
        # 自注意力层
        self.self_attention = ArsenalAttention(config, layer_idx)
        # 前馈神经网络处理
        self.feed_forward = FeedForward(config)
        # 归一化层
        self.input_layer_norm = ArsenalLayerNorm(self.hidden_size, config.norm_eps)
        self.post_attention_layer_norm = ArsenalLayerNorm(self.hidden_size, config.norm_eps)

    def forward(self, hidden_states: torch.Tensor, position_embeddings):
        # 定义残差
        residual = hidden_states
        # 层归一化处理
        hidden_states = self.input_layer_norm(hidden_states)
        # 自注意力机制
        hidden_states, _ = self.self_attention(hidden_states, position_embeddings)
        # 残差连接
        hidden_states = residual + hidden_states
        # 全连接层
        residual = hidden_states
        # 层归一化处理
        hidden_states = self.post_attention_layer_norm(hidden_states)
        # 前馈神经网络
        hidden_states = self.feed_forward(hidden_states)
        # 残差连接
        hidden_states = residual + hidden_states
        return hidden_states


class ArsenalModel(nn.Module):
    """
    Arsenal大语言模型
    """

    def __init__(self, config: ArsenalConfig):
        super(ArsenalModel, self).__init__()
        self.context_length = config.context_length
        self.pad_token_id = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self.num_layers = config.num_layers
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, padding_idx=self.pad_token_id)
        self.layers = nn.ModuleList(
            [
                ArsenalDecoderLayer(config, layer_idx) for layer_idx in range(self.num_layers)
            ]
        )
        self.norm = ArsenalLayerNorm(config.hidden_size, config.norm_eps)
        self.output_head = nn.Linear(config.hidden_size, config.vocab_size)
        freq_cos, freq_sin = compute_rope_freq(config, torch.arange(config.max_position_embedding).unsqueeze(0))
        self.register_buffer("freq_cos", freq_cos, persistent=False)
        self.register_buffer("freq_sin", freq_sin, persistent=False)

    def forward(self, input_ids: Optional[torch.Tensor] = None):
        if input_ids is None:
            raise ValueError("input_ids不能为空")
        _, seq_len = input_ids.shape
        # 词嵌入
        hidden_states = self.embed_tokens(input_ids)
        # 采用旋转位置编码取代-
        position_embeddings = (
            self.freq_cos[:, hidden_states.shape[1]],
            self.freq_sin[:, hidden_states.shape[1]]
        )
        # 循环层处理
        for decoder_layer in self.layers[:self.num_layers]:
            hidden_states = decoder_layer(hidden_states, position_embeddings)
        # 归一化处理
        hidden_states = self.norm(hidden_states)
        # 输出结果(bsz,seq_len,hidden_size)->(bsz,seq_len,vocab_size)
        return self.output_head(hidden_states)


def generate(model: ArsenalModel, idx, temperature=0.0, top_k=None):
    """
    生成下一个词模块
    """
    for _ in range(model.context_length):
        # 每个维度只取最后context_size元素 (bsz,seq_len)
        idx_cond = idx[:, -model.context_length:]
        # 获取结果
        with torch.no_grad():
            logits: torch.Tensor = model(idx_cond)
        # 计算预测值,但是切最后一个 (bsz,seq_len,vocab_size)->(bsz,vocab_size)
        logits = logits[:, -1, :]
        # top K采样
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            top_k_value = top_logits[:, -1]
            logits = torch.where(logits < top_k_value, torch.tensor(float("-inf")).to(logits.device), logits)

        # 温度校正
        if temperature > 0.0:
            logits = logits / temperature
            # 运用softmax归一化
            probs = torch.softmax(logits, dim=-1)
            # 调用随机采样 (bsz,vocab_size)->(batch_size, 1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # 如果未启用采样，选择概率最高的token(bsz,vocab_size)->(batch_size, 1)
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        # 如果出现了结尾字符则停止生成
        if idx_next == model.eos_token_id:
            break
        # 拼接idx (batch_size, seq_len)->(batch_size, seq_len+1)
        idx = torch.cat((idx, idx_next), dim=1)  #
    return idx


def text_to_token_ids(text):
    """
    将文本编码为向量
    """
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids):
    """
    将索引转换为文本
    """
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


if __name__ == '__main__':
    arsenal_model_config = ArsenalConfig(hidden_size=512, num_attention_heads=8, num_layers=8, head_dim=64,
                                         max_position_embedding=512, intermediate_size=2048, context_length=512,
                                         max_train_seq_length=512, epochs=1, learn_rate=0.0005, num_workers=1,
                                         eval_freq=100,
                                         batch_size=32)
    model = ArsenalModel(arsenal_model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(
        torch.load("E:\\ai-env\\model\\small\\arsenal_model.pth", map_location=device, weights_only=True))
    model.eval()
    answer = generate(model, text_to_token_ids("我在北京，我想知道明天会不会下雨。"), temperature=0.0,
                      top_k=100)
    print(token_ids_to_text(answer))

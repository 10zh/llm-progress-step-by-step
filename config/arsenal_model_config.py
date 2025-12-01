class ArsenalConfig:
    def __init__(self, vocab_size=151936, hidden_size=4096, num_attention_heads=32, num_layers=32, attention_bias=True,
                 attention_dropout=0.0, head_dim=128, max_position_embedding=32768, norm_eps=1e-5,
                 intermediate_size=12288, pad_token_id=151643, bos_token_id=151643, eos_token_id=151645,
                 context_length=4096):
        r"""
        以下是Arsenal模型的基础配置参数
        :param vocab_size:  Arsenal模型的词表大小,默认为151936
        :param hidden_size: 隐藏层维度,默认值为4096
        :param num_attention_heads: 注意力头数,默认值为32
        :param num_layers: Transformer中Decoder层数
        :param attention_bias: 训练时的偏置参数,默认为False
        :param attention_dropout: 训练时的丢弃率,默认值为0.0
        :param head_dim: 每个注意力头维度
        :param max_position_embedding: 能够处理的最大位置编码
        :param norm_eps: 层归一化时参数,默认值为1e-5
        :param intermediate_size: 输出时中间层大小
        :param pad_token_id: 填充的词索引位置
        :param bos_token_id: 开始的词索引位置
        :param eos_token_id: 结束的词索引位置
        :param context_length: 上下文长度
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        self.norm_eps = norm_eps
        self.max_position_embedding = max_position_embedding
        self.intermediate_size = intermediate_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.context_length = context_length


__all__ = ["ArsenalConfig"]

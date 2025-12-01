from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('tokenization', trust_remote_code=True, pad_token='<|endoftext|>')


class ArsenalDataset(Dataset):
    def __init__(self, data, max_length, stride):
        """
        构造数据集
        :param data: 数据
        :param max_length: 最大长度
        :param stride: 步幅
        """
        # 输入样例
        self.input_ids = []
        # 输出样例
        self.target_ids = []
        # 文本编码
        token_ids = tokenizer.encode(data)
        # 使用滑动窗口构建
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(data, batch_size=4, max_length=4096,
                         stride=512, shuffle=True, drop_last=True,
                         num_workers=0):
    # 创建数据集
    dataset = ArsenalDataset(data, max_length, stride)
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader
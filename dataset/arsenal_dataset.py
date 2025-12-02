from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('tokenization', trust_remote_code=True, pad_token='<|endoftext|>')


def custom_padding(
        batch,
        pad_token_id=151643
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item) + 1 for item in batch)

    # Pad and prepare inputs
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = (
                new_item + [pad_token_id] *
                (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    return inputs_lst, targets_lst


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
        # 分词样例
        tokens = []
        for item in data:
            # 文本编码
            token_ids = tokenizer.encode(item + "<|endoftext|>")
            # 如果小于则添加到token中
            if len(token_ids) < max_length:
                tokens = tokens + token_ids
                continue
            # 使用滑动窗口构建
            for i in range(0, len(tokens) - max_length, stride):
                input_chunk = tokens[i:i + max_length]
                target_chunk = tokens[i + 1: i + max_length + 1]
                self.input_ids.append(torch.tensor(input_chunk))
                self.target_ids.append(torch.tensor(target_chunk))
            # 重置tokens
            tokens = []

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(data, batch_size=4, max_length=4096,
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

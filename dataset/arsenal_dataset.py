import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('tokenization', trust_remote_code=True, pad_token='<|endoftext|>')


class ArsenalDataset(Dataset):
    def __init__(self, data, pad_token_id=151643):
        """
        构造数据集
        :param data: 数据
        """
        # 输入样例
        self.input_ids = []
        # 输出样例
        self.target_ids = []
        # 分词样例
        tokens = []
        for item in data:
            # 先把所有数据编码
            token_ids = tokenizer.encode(item)
            # 添加到所有的分词样例中
            tokens.append(token_ids)
        # 找到批量中最长的那部分数据
        batch_max_length = max(len(item) + 1 for item in tokens)
        # 遍历批量样本数据
        for item in tokens:
            # 复制一份数据
            new_item = item.copy()
            # 填充结尾token
            new_item += [pad_token_id]
            # 填充到batch_max_length
            padded = (
                    new_item + [pad_token_id] * (batch_max_length - len(new_item))
            )
            inputs = torch.tensor(padded[:-1])
            targets = torch.tensor(padded[1:])
            self.input_ids.append(inputs)
            self.target_ids.append(targets)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(data, batch_size=4, shuffle=True, drop_last=True,
                      num_workers=0):
    # 创建数据集
    dataset = ArsenalDataset(data)
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

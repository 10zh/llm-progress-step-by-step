import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('../tokenization', trust_remote_code=True, pad_token='<|endoftext|>')


class ArsenalDataset(Dataset):
    def __init__(self, data, max_length):
        """
        构造数据集
        :param data: 数据
        """
        # 输入样例
        self.input_ids = []
        # 输出样例
        self.target_ids = []
        # 文本最大长度
        self.max_length = max_length
        # 遍历数据
        for item in data:
            token_ids = tokenizer.encode(item, truncation=True, max_length=self.max_length, stride=1,
                                         padding='max_length')
            inputs = torch.tensor(token_ids[:-1])
            targets = torch.tensor(token_ids[1:])
            self.input_ids.append(inputs)
            self.target_ids.append(targets)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(data, batch_size=4, shuffle=True, drop_last=True,
                      num_workers=0, max_length=4096):
    # 创建数据集
    dataset = ArsenalDataset(data, max_length)
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

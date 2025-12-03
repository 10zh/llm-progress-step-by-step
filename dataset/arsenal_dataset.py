import torch
from torch.utils.data import DataLoader, Dataset


class ArsenalDataset(Dataset):
    def __init__(self, tokenizer, data, max_length):
        """
        构造数据集
        :param data: 数据
        """
        # 分词器
        self.tokenizer = tokenizer
        # 输入样例
        self.input_ids = []
        # 输出样例
        self.target_ids = []
        # 文本最大长度
        self.max_length = max_length
        # 遍历数据
        for (i, item) in enumerate(data):
            token_ids = tokenizer.encode(item, truncation=True, max_length=self.max_length, stride=1,
                                         padding='max_length')
            inputs = torch.tensor(token_ids[:-1])
            targets = torch.tensor(token_ids[1:])
            if i % 50000 == 0:
                print(f"已批量处理的数据:{i}")
            self.input_ids.append(inputs)
            self.target_ids.append(targets)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(data, tokenizer, batch_size=32, shuffle=True, drop_last=True,
                      num_workers=0, max_length=4096):
    # 创建数据集
    dataset = ArsenalDataset(tokenizer=tokenizer, data=data, max_length=max_length)
    # 创建数据加载器
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

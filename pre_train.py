import json
import os
import time

import torch
from transformers import AutoTokenizer

from arsenal_basic_model import ArsenalModel
from config.arsenal_model_config import ArsenalConfig
from dataset.arsenal_dataset import create_dataloader

tokenizer = AutoTokenizer.from_pretrained('tokenization', trust_remote_code=True, pad_token='<|endoftext|>')


def calc_batch_loss(input_batch: torch.Tensor, target_batch: torch.Tensor, model: ArsenalModel, device):
    # 使用GPU
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # 计算值(bsz,seq_len,vocab_size)
    logits = model(input_batch)
    # 使用交叉熵计算损失
    # logits.flatten(0, 1):(bsz,seq_len,vocab_size)->(bsz*seq_len,vocab_size)
    # target_batch.flatten():(bsz,seq_len)->(bsz*seq_len)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loader_loss(data_loader, model: ArsenalModel, device, num_batches=None):
    """计算某批次的损失"""
    # 定义总损失
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # 防止批次超过数据加载器的总批次
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_batch_loss(input_batch, target_batch, model, device)
            total_loss += loss
        else:
            break
    return total_loss / num_batches


def evaluate_arsenal_model(model: ArsenalModel, train_loader, device, eval_iter):
    """
    评估模型
    :param model: 模型
    :param train_loader: 训练集
    :param device: 设备
    :param eval_iter: 验证批次
    :return: 训练集损失和验证集损失
    """
    # 模型启用推理模式
    model.eval()
    # 防止梯度更新
    with torch.no_grad():
        # 获取训练数据损失
        loss = calc_loader_loss(train_loader, model, device, num_batches=eval_iter)
    # 重新设置为训练模式
    model.train()
    return loss


def train_arsenal_model(model: ArsenalModel, train_loader, num_epochs, device, optimizer, eval_freq):
    """
    训练模型
    :param model: 模型
    :param train_loader: 训练集
    :param num_epochs: 训练轮数
    :param device: 涉笔信息
    :param optimizer: 梯度优化器
    :param eval_freq: 每多少次评估损失
    :return: 总训练集损失,验证集损失
    """
    # 定义训练验证总损失
    train_losses = []
    # 定义总训练步数
    global_step = 0
    # 获取训练集大小
    iter_length = len(train_loader)
    # 训练轮数
    for epoch in range(num_epochs):
        # 设置模型为训练模式
        model.train()
        # 遍历数据加载器
        for step, (input_batch, target_batch) in enumerate(train_loader):
            # 清空梯度
            optimizer.zero_grad()
            # 计算损失函数
            loss = calc_batch_loss(input_batch, target_batch, model, device)
            # 记录损失
            current_loss = loss.item()
            train_losses.append(current_loss)
            # 基于损失函数反向传播优化
            loss.backward()
            # 更新权重
            optimizer.step()
            # 训练次数+1
            global_step += 1
            # 多少次样本或最后一轮输出损失
            if global_step % eval_freq == 0 or step == iter_length - 1:
                print(
                    f"训练轮数:{epoch + 1},训练次数:{global_step},训练集损失:{current_loss:.6f}")

    # 训练完成返回总损失
    return train_losses


def read_jsonl_content_generator(directory_path, text_key):
    """
        生成器版本，逐行读取，节省内存
    """
    contents = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.jsonl'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                content = data.get(text_key)
                                if content:
                                    # 拼接字符
                                    contents.append(content)
                            except json.JSONDecodeError:
                                continue
    return contents


def read_json_config_file():
    """
    读取配置文件,返回模型配置对象
    """
    with open('./model_config.json', 'r') as file:
        config_data = json.load(file)
        arsenal_config = ArsenalConfig()
        for key, value in config_data.items():
            if value is not None and hasattr(arsenal_config, key):
                setattr(arsenal_config, key, value)
        return arsenal_config


if __name__ == '__main__':
    # 固定随机种子数
    torch.manual_seed(123)
    # 初始化模型
    train_model_config = read_json_config_file()
    train_model = ArsenalModel(train_model_config)
    # 获取设备信息
    train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model.to(train_device)
    # 使用AdamW梯度优化器
    train_optimizer = torch.optim.AdamW(train_model.parameters(), lr=train_model_config.learn_rate, weight_decay=0.1)
    # 训练开始时间
    start_time = time.time()
    # 构造训练集
    train_total_losses = train_arsenal_model(
        train_model,
        create_dataloader(read_jsonl_content_generator("./dataset/data/train", "content"), tokenizer=tokenizer,
                          num_workers=train_model_config.num_workers,
                          max_length=train_model_config.max_train_seq_length),
        train_model_config.epochs, train_device, train_optimizer
        , eval_freq=train_model_config.eval_freq
    )
    torch.save(train_model.state_dict(), "arsenal_model.pth")
    # 获取训练结束时间
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"训练完成所花费时间:{execution_time_minutes:.2f}分钟.")

import itertools


def copy_first_50000_lines_islice(input_file, output_file, n=50000):
    """使用itertools.islice，内存高效"""
    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            # 直接读取前n行
            for line in itertools.islice(f_in, n):
                f_out.write(line)
    print(f"已成功复制前 {n} 行到 {output_file}")


# 使用
if __name__ == '__main__':
    copy_first_50000_lines_islice('E:\\ai-env\dataset\pretrain_hq.jsonl', '/dataset/data/train/pre_train_small.jsonl')

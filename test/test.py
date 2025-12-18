import torch

if __name__ == '__main__':
    x = torch.ones(2, 2, 4)
    print(x[..., x.shape[-1] // 2:].shape)

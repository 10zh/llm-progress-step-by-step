import torch

if __name__ == '__main__':
    freq = 1.0 / 100 ** torch.arange(0, 10, 2)
    a = torch.ones(2, 1, 3, 4) * torch.ones(2, 2, 3, 4)
    x = torch.ones(2, 3, 4)
    print(a.shape)
    print(torch.ones(1,10)[:, None, :].shape)
    print(torch.arange(10).unsqueeze(0).shape)
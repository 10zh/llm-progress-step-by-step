import torch

if __name__ == '__main__':
    freq = 1.0 / 100 ** torch.arange(0, 10, 2)
    a = torch.ones(2, 1, 3, 4) * torch.ones(2, 2, 3, 4)
    x = torch.ones(2, 3, 4)
    print(x[..., :x.shape[-1] // 2].shape)
    print(a.shape)
    print(freq)
    print(torch.ones(2, 3)[:, None, :].shape)
    print(torch.ones(2, 3).unsqueeze(1).shape)
    print(torch.cat((torch.ones(2, 3)[:, None, :], torch.ones(2, 3)[:, None, :]), dim=-1).shape)
    print(freq[None, :, None].shape)
    print(freq[None, :, None].expand(3, -1, 1).shape)

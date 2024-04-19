import torch
a=torch.randn(10)
values=torch.randn(10)
indices=torch.LongTensor()
torch.max(a, 0, out=(values, indices))
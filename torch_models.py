import torch


class mlp_baseline(torch.nn.Module):
    def __init__(self,inputsize, depth, width):
        super().__init__()
        self.inp = torch.nn.Linear(inputsize, width)
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(width, width),
                torch.nn.Tanh(),
            ) for _ in range(depth)
        ])
        self.out = torch.nn.Linear(width, 1)

    def forward(self, x):
        x = self.inp(x)
        for block in self.blocks:
            x = block(x)
        return self.out(x)
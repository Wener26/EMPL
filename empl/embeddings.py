import torch
from torch import nn


class RelativePhasePrototype(nn.Module):
    def __init__(self, seq_len, d_t, n_stages=3):
        super().__init__()
        self.dictionary = nn.Embedding(n_stages, d_t)
        u = torch.arange(seq_len, dtype=torch.float) / (seq_len - 1)
        stage = torch.clamp(torch.floor(n_stages * u).long(), max=n_stages - 1)
        self.register_buffer("stage", stage)

    def forward(self, batch_size):
        e = self.dictionary(self.stage).mean(dim=0)
        return e.unsqueeze(0).expand(batch_size, -1)


class PhysiologyEmbedding(nn.Module):
    def __init__(self, in_channels, d_t, n_virtual=8, smooth_len=51):
        super().__init__()
        self.project = nn.Conv1d(in_channels, n_virtual, kernel_size=1)
        self.smooth = nn.AvgPool1d(smooth_len, stride=1, padding=smooth_len // 2, count_include_pad=False)
        self.norm = nn.InstanceNorm1d(n_virtual, affine=False)
        self.mlp = nn.Sequential(nn.Linear(n_virtual, 32), nn.ReLU(), nn.Linear(32, d_t), nn.Tanh())

    def forward(self, x):
        p = self.smooth(self.project(x) ** 2)
        z = self.norm(torch.log(p + 1e-6))
        return self.mlp(z.mean(dim=2))


class HybridTemporalEmbedding(nn.Module):
    def __init__(self, in_channels, seq_len, d_t, alpha=0.3, n_stages=3):
        super().__init__()
        self.proto = RelativePhasePrototype(seq_len, d_t, n_stages)
        self.phys = PhysiologyEmbedding(in_channels, d_t)
        self.alpha = alpha

    def forward(self, x):
        return self.proto(x.shape[0]) + self.alpha * self.phys(x)

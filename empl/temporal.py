import math

import torch
import torch.nn.functional as F
from torch import nn

from .meta import balanced_coefficients


class MetaCausalConv1d(nn.Module):
    def __init__(self, channels, kernel_size, d_t, d_s, n_groups, tau=1.0):
        super().__init__()
        self.channels, self.kernel_size, self.n_groups, self.tau = channels, kernel_size, n_groups, tau
        self.gate = nn.Linear(d_s, d_t)
        self.weight_pool = nn.Parameter(torch.empty(d_t, channels, channels, kernel_size))
        self.bias_pool = nn.Parameter(torch.zeros(d_t, channels))
        nn.init.xavier_uniform_(self.weight_pool)

    def coefficients(self, e_t, e_s):
        q = e_t * torch.sigmoid(self.gate(e_s.mean(dim=0, keepdim=True)))
        return balanced_coefficients(q, self.n_groups, self.tau)

    def forward(self, x, a):
        b, c, _ = x.shape
        w = torch.einsum("bd,doik->boik", a, self.weight_pool)
        bias = torch.einsum("bd,do->bo", a, self.bias_pool)
        xp = F.pad(x, (self.kernel_size - 1, 0)).reshape(1, b * c, -1)
        out = F.conv1d(xp, w.reshape(b * self.channels, c, self.kernel_size), bias.reshape(-1), groups=b)
        return out.view(b, self.channels, -1)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_head=2, key_dim=8, dropout=0.5):
        super().__init__()
        self.n_head, self.key_dim = n_head, key_dim
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * n_head * key_dim)
        self.out = nn.Linear(n_head * key_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, l, _ = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, l, 3, self.n_head, self.key_dim).permute(2, 0, 3, 1, 4)
        attn = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.key_dim), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, l, self.n_head * self.key_dim)
        return x + self.dropout(self.out(out))


class WindowBranch(nn.Module):
    def __init__(self, meta_conv, d_model, n_head=2, key_dim=8, dropout_attn=0.5, dropout_conv=0.3):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, n_head, key_dim, dropout_attn)
        self.meta_conv = meta_conv
        self.norm = nn.BatchNorm1d(d_model)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout_conv)

    def forward(self, x, a):
        h = self.attention(x).transpose(1, 2)
        o = h + self.dropout(self.act(self.norm(self.meta_conv(h, a))))
        return o.mean(dim=2)


class MetaTemporalEncoder(nn.Module):
    def __init__(self, n_windows, d_model, kernel_size, d_t, d_s, n_groups, tau=1.0,
                 n_head=2, key_dim=8, dropout_attn=0.5, dropout_conv=0.3):
        super().__init__()
        self.n_windows = n_windows
        self.meta_conv = MetaCausalConv1d(d_model, kernel_size, d_t, d_s, n_groups, tau)
        self.windows = nn.ModuleList([
            WindowBranch(self.meta_conv, d_model, n_head, key_dim, dropout_attn, dropout_conv)
            for _ in range(n_windows)])

    def forward(self, seq, e_t, e_s):
        a = self.meta_conv.coefficients(e_t, e_s)
        length = seq.shape[1] - self.n_windows + 1
        outs = [w(seq[:, i:i + length], a) for i, w in enumerate(self.windows)]
        return torch.stack(outs, dim=0).mean(dim=0)

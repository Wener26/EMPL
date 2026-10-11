import torch
from torch import nn

from .embeddings import HybridTemporalEmbedding
from .meta import group_orthogonality_loss
from .spatial import MetaSpatialEncoder
from .temporal import MetaTemporalEncoder


class MaxNormConv2d(nn.Conv2d):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, F1=16, D=2, kernel_length=64, pool1=8, pool2=7, dropout=0.3):
        super().__init__()
        self.temporal = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.spatial = MaxNormConv2d(F1, F1 * D, (in_channels, 1), groups=F1, bias=False, max_norm=1.0)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, pool1))
        self.conv = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F1 * D)
        self.pool2 = nn.AvgPool2d((1, pool2))
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.out_channels = F1 * D

    def forward(self, x):
        x = self.bn1(self.temporal(x.unsqueeze(1)))
        x = self.dropout(self.pool1(self.act(self.bn2(self.spatial(x)))))
        x = self.dropout(self.pool2(self.act(self.bn3(self.conv(x)))))
        return x.squeeze(2)


class EMPL(nn.Module):
    def __init__(self, in_channels=22, seq_len=1125, n_classes=4, n_windows=5, n_stages=3,
                 d_t=64, d_s=64, alpha=0.3, tau=1.0, n_groups=8, J=2, kernel_length=4, hidden_dim=32):
        super().__init__()
        self.temporal_embedding = HybridTemporalEmbedding(in_channels, seq_len, d_t, alpha, n_stages)
        self.spatial_embedding = nn.Embedding(in_channels, d_s)
        self.cnn = ConvBlock(in_channels)
        d_model = self.cnn.out_channels
        self.temporal_encoder = MetaTemporalEncoder(n_windows, d_model, kernel_length, d_t, d_s, n_groups, tau)
        with torch.no_grad():
            n_steps = self.cnn(torch.zeros(1, in_channels, seq_len)).shape[-1]
        self.node_pool = nn.AdaptiveAvgPool1d(n_steps)
        self.spatial_encoder = MetaSpatialEncoder(d_s, d_t, 1, hidden_dim, J, n_groups, tau)
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(d_model + hidden_dim, n_classes))

    def meta_pools(self):
        return [self.temporal_encoder.meta_conv.weight_pool, self.spatial_encoder.pool]

    def group_diversity_loss(self):
        pools = self.meta_pools()
        return sum(group_orthogonality_loss(p, self.spatial_encoder.n_groups) for p in pools) / len(pools)

    def forward(self, x, return_features=False):
        e_t = self.temporal_embedding(x)
        e_s = self.spatial_embedding.weight
        seq = self.cnn(x).transpose(1, 2)
        f_t = self.temporal_encoder(seq, e_t, e_s)
        f_s = self.spatial_encoder(self.node_pool(x), e_s, e_t)
        feat = torch.cat([f_t, f_s], dim=1)
        logits = self.classifier(feat)
        if return_features:
            return logits, feat
        return logits

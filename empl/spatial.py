import torch
from torch import nn

from .meta import balanced_coefficients


class MetaGCRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, J=2):
        super().__init__()
        self.input_dim, self.hidden_dim, self.J = input_dim, hidden_dim, J

    def diffusion(self, x, A, W):
        out, A_pow = 0, torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
        for j in range(self.J):
            if j > 0:
                A_pow = A_pow @ A
            out = out + torch.matmul((A_pow @ x).unsqueeze(2), W[:, :, j]).squeeze(2)
        return out

    def forward(self, x, h_prev, A, params):
        W_r, b_r, W_u, b_u, W_c, b_c = params
        xh = torch.cat([x, h_prev], dim=-1)
        r = torch.sigmoid(self.diffusion(xh, A, W_r) + b_r)
        u = torch.sigmoid(self.diffusion(xh, A, W_u) + b_u)
        c = torch.tanh(self.diffusion(torch.cat([x, r * h_prev], dim=-1), A, W_c) + b_c)
        return u * h_prev + (1 - u) * c


class MetaSpatialEncoder(nn.Module):
    def __init__(self, d_s, d_t, input_dim, hidden_dim, J=2, n_groups=8, tau=1.0):
        super().__init__()
        self.input_dim, self.hidden_dim, self.J, self.n_groups, self.tau = input_dim, hidden_dim, J, n_groups, tau
        self.gate = nn.Linear(d_t, d_s)
        self.weight_size = J * (input_dim + hidden_dim) * hidden_dim
        self.pool = nn.Parameter(torch.randn(d_s, 3 * (self.weight_size + hidden_dim)) * 0.1)
        self.cell = MetaGCRU(input_dim, hidden_dim, J)

    def adjacency(self, e_s):
        return torch.softmax(torch.relu(e_s @ e_s.t()), dim=-1)

    def node_parameters(self, e_s, e_t):
        q = e_s.unsqueeze(0) * torch.sigmoid(self.gate(e_t)).unsqueeze(1)
        theta = balanced_coefficients(q, self.n_groups, self.tau) @ self.pool
        b, n, _ = theta.shape
        params, off = [], 0
        for _ in range(3):
            params.append(theta[:, :, off:off + self.weight_size].reshape(
                b, n, self.J, self.input_dim + self.hidden_dim, self.hidden_dim))
            off += self.weight_size
            params.append(theta[:, :, off:off + self.hidden_dim])
            off += self.hidden_dim
        return tuple(params)

    def forward(self, nodes, e_s, e_t):
        A = self.adjacency(e_s)
        params = self.node_parameters(e_s, e_t)
        h = torch.zeros(nodes.shape[0], nodes.shape[1], self.hidden_dim, device=nodes.device, dtype=nodes.dtype)
        for t in range(nodes.shape[2]):
            h = self.cell(nodes[:, :, t].unsqueeze(-1), h, A, params)
        return h.mean(dim=1)

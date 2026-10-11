import torch


def balanced_coefficients(q, n_groups, tau=1.0):
    k = q.shape[-1]
    m = k // n_groups
    top, idx = q.reshape(q.shape[:-1] + (n_groups, m)).max(dim=-1)
    idx = idx + torch.arange(n_groups, device=q.device) * m
    coef = torch.zeros_like(q)
    coef.scatter_(-1, idx, torch.softmax(top / tau, dim=-1))
    return coef


def group_orthogonality_loss(pool, n_groups):
    k = pool.shape[0]
    mu = pool.reshape(n_groups, k // n_groups, -1).mean(dim=1)
    mu = mu / (mu.norm(dim=1, keepdim=True) + 1e-8)
    s = mu @ mu.t()
    eye = torch.eye(n_groups, device=pool.device, dtype=pool.dtype)
    return ((s - eye) ** 2).sum() / n_groups ** 2

import torch
import torch.nn as nn

class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))

    def sample_h(self, v):
        p_h = torch.sigmoid(torch.matmul(v, self.W.t()) + self.h_bias)
        return p_h.bernoulli(), p_h

    def sample_v(self, h):
        p_v = torch.sigmoid(torch.matmul(h, self.W) + self.v_bias)
        return p_v.bernoulli(), p_v

    def gibbs_sampling(self, v, steps=1):
        for _ in range(steps):
            h, _ = self.sample_h(v)
            v, _ = self.sample_v(h)
        return v

    def forward(self, v):
        h, _ = self.sample_h(v)
        v_recon, _ = self.sample_v(h)
        return v_recon

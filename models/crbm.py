import torch
import torch.nn as nn

class CRBM(nn.Module):
    def __init__(self, n_visible, n_hidden, n_cond):
        super(CRBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.U = nn.Parameter(torch.randn(n_hidden, n_cond) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))

    def sample_h(self, v, c):
    # Just ensure all on same device (usually v, c are already moved outside)
        activation = torch.matmul(v, self.W.t()) + torch.matmul(c, self.U.t()) + self.h_bias
        p_h = torch.sigmoid(activation)
        return p_h.bernoulli(), p_h


    def sample_v(self, h):
        p_v = torch.sigmoid(torch.matmul(h, self.W) + self.v_bias)
        return p_v.bernoulli(), p_v

    def gibbs_sampling(self, v, c, steps=1):
        device = v.device  # Get the device of the input tensor
        for _ in range(steps):
            h, _ = self.sample_h(v.to(device), c.to(device))  # Ensure c is on the same device
            v, _ = self.sample_v(h)
        return v

    def forward(self, v, c):
        h, _ = self.sample_h(v, c)
        v_recon, _ = self.sample_v(h)
        return v_recon

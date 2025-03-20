import torch
import torch.nn as nn
import torch.nn.functional as F

class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        """
        Restricted Boltzmann Machine
        Args:
            n_visible: Number of visible units
            n_hidden: Number of hidden units
        """
        super(RBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        
        # Initialize parameters
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))

    def sample_h(self, v):
        """Sample hidden units given visible units"""
        activation = F.linear(v, self.W.t(), self.h_bias)
        p_h = torch.sigmoid(activation)
        return torch.bernoulli(p_h), p_h

    def sample_v(self, h):
        """Sample visible units given hidden units"""
        activation = F.linear(h, self.W, self.v_bias)
        p_v = torch.sigmoid(activation)
        return torch.bernoulli(p_v), p_v

    def forward(self, v, k=1):
        """Contrastive Divergence with k steps"""
        # Positive phase
        h_pos, _ = self.sample_h(v)
        
        # Negative phase (Gibbs sampling)
        v_neg = v.clone()
        for _ in range(k):
            h_neg, _ = self.sample_h(v_neg)
            v_neg, _ = self.sample_v(h_neg)
        
        return h_pos, v_neg

    def free_energy(self, v):
        """Calculate free energy for gradient calculation"""
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v, self.W.t(), self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()

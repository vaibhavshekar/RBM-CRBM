import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

class CRBM(nn.Module):
    def __init__(self, n_visible, n_hidden, n_cond):
        """
        Conditional Restricted Boltzmann Machine
        Args:
            n_visible: Number of visible units
            n_hidden: Number of hidden units
            n_cond: Number of conditional units
        """
        super(CRBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_cond = n_cond
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible))
        self.U = nn.Parameter(torch.randn(n_hidden, n_cond) if n_cond else None)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

    def sample_h(self, v, c):
        """Sample hidden units given visible and conditional units"""
        activation = F.linear(v, self.W, self.h_bias) + F.linear(c, self.U)
        activation = activation.clamp(-80, 80)  # Prevent overflow
        p_h = torch.sigmoid(activation)
        return torch.bernoulli(p_h), p_h

    def sample_v(self, h):
        """Sample visible units given hidden units"""
        activation = F.linear(h, self.W.t(), self.v_bias)
        p_v = torch.sigmoid(activation)
        return torch.bernoulli(p_v), p_v

    def forward(self, v, c, k=1):
        """Contrastive Divergence with k steps"""
        # Positive phase
        h_pos, _ = self.sample_h(v, c)
        
        # Negative phase (Gibbs sampling)
        v_neg = v.clone()
        for _ in range(k):
            h_neg, _ = self.sample_h(v_neg, c)
            v_neg, _ = self.sample_v(h_neg)
        
        return h_pos, v_neg

    def free_energy(self, v, c):
        """Calculate free energy for gradient calculation"""
        vbias_term = v.mv(self.v_bias)  # Shape: (batch_size,)
    
        # Ensure correct shapes for linear transformations
        wx_b = F.linear(v, self.W, self.h_bias) + F.linear(c, self.U)
        wx_b = wx_b.clamp(-80, 80)
        
        hidden_term = wx_b.exp().add(1).log().sum(1)  # Shape: (batch_size,)
        
        return (-hidden_term - vbias_term).mean()


    def gibbs_sampling(self, v, c, steps=1):
        device = v.device # Get the device of the input tensor
        for _ in range(steps):
            h, _ = self.sample_h(v.to(device), c.to(device)) # Ensure c is on the same device
            v, _ = self.sample_v(h)
        return v

    def generate(self, c, steps=100):
        """Generate samples from conditional input"""
        v = torch.rand((1, self.n_visible)).to(c.device)
        for _ in range(steps):
            h, _ = self.sample_h(v, c)
            v, _ = self.sample_v(h)
        return v

    def contrastive_divergence_loss(self, v, c, k=1):
        # Positive phase
        h_prob_pos = torch.sigmoid(torch.matmul(v, self.W.t()) + torch.matmul(c, self.U.t()) + self.h_bias)
        positive_grad = torch.matmul(h_prob_pos.t(), v)

        # Negative phase (reconstructed)
        v_neg = self.gibbs_sampling(v, c, steps=k)
        h_prob_neg = torch.sigmoid(torch.matmul(v_neg, self.W.t()) + torch.matmul(c, self.U.t()) + self.h_bias)
        negative_grad = torch.matmul(h_prob_neg.t(), v_neg)

        # Contrastive divergence loss: difference in gradients
        loss = torch.mean((positive_grad - negative_grad)**2)  # This is a squared difference between phases

        return loss

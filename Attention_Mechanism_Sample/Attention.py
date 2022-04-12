import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import FCNet

class Attention(nn.Module):
    def __init__(self, v_dim, q_dim, k, dropout=0.2):
        super(Attention, self).__init__()

        self.v_proj = FCNet([v_dim, k])
        self.q_proj = FCNet([q_dim, k])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(k, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, T, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, T, _ = v.size()
        v_proj = self.v_proj(v) # [batch, T, d]
        q_proj = self.q_proj(q)
        q_proj = q_proj.unsqueeze(2)
        q_proj = q_proj.repeat(1, T, 1).reshape((batch, 16, 512))
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits
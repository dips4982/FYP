import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SimpleClassifier, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        logits = self.main(x)
        return logits
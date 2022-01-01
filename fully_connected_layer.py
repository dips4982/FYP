import torch
from torch import nn

class FCLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # define the layers
        self.layers = nn.Sequential(
            nn.Linear(4, 16)
        )
    
    def forward(self, x):
        # forward pass
        x = torch.softmax(self.layers(x))
        return x

# instantiate the model
model = FCLayer()

# print model architecture
print(model)
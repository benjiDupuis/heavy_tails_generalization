import numpy as np 
import torch
import torch.nn as nn

class LinearModel(nn.Module):

    def __init__(self, 
                 input_dim: int = 10,
                 bias: bool = False,
                 n_classes: int = 2):
        super(LinearModel, self).__init__()
        
        self.input_dim: int = input_dim
        self.bias: bool = bias
        self.layer = nn.Linear(self.input_dim, n_classes, bias = self.bias)

    @torch.no_grad()
    def get_vector(self) -> torch.Tensor:
        # TODO: implement this
        return self.layer.weight.data.detach().to("cpu")

    def forward(self, x):
        x = x.view(x.size(0), self.input_dim) # I don't know why we have this
        x = self.layer(x)
        return x
    
    @torch.no_grad()
    def add_noise(self, noise: torch.Tensor):
        # This methods add a noise to the parameters of the networks
        assert not self.bias, "For now we don't handle biases" 
        # self.layer.weight.data = self.layer.weight.data + noise
        self.layer.weight.add_(noise)

    @torch.no_grad()
    def initialization(self, w: torch.Tensor):
        assert not self.bias, "For now we don't handle biases" 
        assert self.layer.weight.data.shape == w.shape,\
              (self.layer.weight.data.shape, w.shape)
        self.layer.weight.data = w
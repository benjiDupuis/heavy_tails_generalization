import torch
import torch.nn as nn

class NoisyGDModel(nn.Module):

    input_dim: int
    n_classes: int = 2
    bias: bool = False

    @torch.no_grad()
    def params_number(self) -> int:
        total = 0
        for param in self.parameters():
            total += param.numel()
        return total  

    @torch.no_grad()
    def gradient_l2_squared_norm(self) -> float:
        total_norm = 0.
        for param in self.parameters():
            total_norm += param.grad.data.norm(2).item()**2
        return float(total_norm)   
    
    @torch.no_grad()
    def add_noise(self, w: torch.Tensor):

        assert w.ndim == 1, w.ndim
        assert w.size()[0] == self.params_number(), \
            (w.size()[0], self.params_number())

        count = 0
        for param in self.parameters():
            param_len = param.numel()
            param_size = param.size()
            param.add_(w[count:(count+param_len)].reshape(param_size))
            count += param_len

    @staticmethod
    def num_params() -> int:
        pass
     
    @torch.no_grad()
    def initialization(self, w: torch.Tensor):
        # TODO: implement this
        pass


class LinearModel(NoisyGDModel):

    def __init__(self, 
                 input_dim: int = 10,
                 bias: bool = False,
                 n_classes: int = 2):
        super(LinearModel, self).__init__()
        
        self.input_dim: int = input_dim
        self.bias: bool = bias
        self.num_classes: int = n_classes

        self.layer = nn.Linear(self.input_dim, n_classes, bias = self.bias)

    @staticmethod
    def num_params(input_dim: int,
                     n_classes: int,
                     bias: bool = False) -> int:
        linear_num = input_dim * n_classes
        if bias:
            return linear_num + n_classes
        else:
            return linear_num

    @torch.no_grad()
    def get_vector(self) -> torch.Tensor:
        # TODO: implement this
        return self.layer.weight.data.detach().to("cpu")

    def forward(self, x):
        x = x.view(x.size(0), self.input_dim) # this flatten potential images
        x = self.layer(x)
        return x

    @torch.no_grad()
    def initialization(self, w: torch.Tensor):
        assert not self.bias, "For now we don't handle biases" 
        assert self.layer.weight.data.shape == w.shape,\
              (self.layer.weight.data.shape, w.shape)
        self.layer.weight.data = w 


class FCNN(NoisyGDModel):

    def __init__(self, 
                 depth: int = 5,
                  width: int = 50,
                    input_dim: int = 10,
                    n_classes: int = 2,
                    bias: bool = False):
        super(FCNN, self).__init__()

        assert depth >= 1,\
                "depth should be at least 1, for 0 depth, use LinearModel"

        self.input_dim: int = input_dim
        self.width: int = width
        self.depth: int = depth
        self.bias: bool = bias
        self.n_classes: int = n_classes

        layers = self.get_layers()

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.width, bias=self.bias),
            nn.ReLU(inplace=True),
            *layers,
            nn.Linear(self.width, self.n_classes, bias=self.bias),
        )

    @staticmethod
    def num_params(depth: int = 5,
                  width: int = 50,
                    input_dim: int = 10,
                    n_classes: int = 2,
                    bias: bool = False) -> int:
        
        linear_num = (depth - 1) * (width**2) + \
                            input_dim * width + \
                            n_classes * width
        if bias:
            return linear_num + n_classes + depth * width
        else:
            return linear_num

    def get_layers(self):
        layers = []
        for _ in range(self.depth - 1):
            layers.append(nn.Linear(self.width, self.width, bias=self.bias))
            layers.append(nn.ReLU())
        return layers

    def forward(self, x):
        x = x.view(x.size(0), self.input_dim)
        x = self.fc(x)
        return x


def fcnn_num_params(input_dim: int,
         width: int = None,
         depth: int = 2,
         bias: bool = False,
         n_classes = 2) -> int:

    if depth == 0:

        return LinearModel.num_params(
            input_dim = input_dim,
            bias = bias,
            n_classes = n_classes
        )
    
    else:
        assert width is not None, "width must be provided for non linear models"

        return FCNN.num_params(
            depth = depth,
            width = width,
            input_dim = input_dim,
            n_classes = n_classes,
            bias = bias
        )
    

def fcnn(input_dim: int,
         width: int = None,
         depth: int = 2,
         bias: bool = False,
         n_classes = 2) -> NoisyGDModel:
    
    if depth == 0:

        return LinearModel(
            input_dim = input_dim,
            bias = bias,
            n_classes = n_classes
        )
    
    else:
        assert width is not None, "width must be provided for non linear models"

        return FCNN(
            depth = depth,
            width = width,
            input_dim = input_dim,
            n_classes = n_classes,
            bias = bias
        )



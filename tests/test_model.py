import torch

from last_point.model import fcnn, fcnn_num_params


DIMENSION = 2 # Input dimension for tests
CLASSES = 2 # Number of classes for tests
N = 5


def test_linear_model():

    device = torch.device("cpu")

    model = fcnn(input_dim=DIMENSION,\
                    depth=0,
                 n_classes=CLASSES).to(device)
    assert model.params_number() == fcnn_num_params(input_dim=DIMENSION,\
                                                    depth=0,
                                                     n_classes=CLASSES)
    n_params = model.params_number()
    assert n_params == 4, (n_params, 4)

    # put the weights to 0
    model.layer.weight.data = torch.zeros(2,2)

    # Here we test that the squared gradient norm computations works well.
    input_tensor = torch.ones(N, DIMENSION)
    output = model(input_tensor)
    assert output.shape == (N, CLASSES), output.shape

    assert output.pow(2).sum() == 0.
    
    loss = (output - 1.).pow(2).sum() / N
    loss.backward()

    gradient_norm = model.gradient_l2_squared_norm()
    assert gradient_norm == 16., (gradient_norm, 16.)

    






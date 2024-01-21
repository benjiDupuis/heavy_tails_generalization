from loguru import logger

from data.dataset import get_full_batch_data


def test_load_all_mnist():

    (x_train, y_train, x_test, y_test) = get_full_batch_data("mnist", \
                                "~/data",
                                 subset_percentage=1.)
    
    assert x_train.ndim == 4, x_train.ndim
    assert x_train.shape[0] == 60000, x_train.shape
    assert x_train.shape[1] == 1, x_train.shape
    assert x_train.shape[2] == 28, x_train.shape
    assert x_train.shape[3] == 28, x_train.shape

    assert y_train.ndim == 1, y_train.ndim
    assert y_train.shape[0] == 60000, y_train.shape

    assert x_test.ndim == 4, x_test.ndim
    assert x_test.shape[0] == 10000, x_test.shape
    assert x_test.shape[1] == 1, x_test.shape
    assert x_test.shape[2] == 28, x_test.shape
    assert x_test.shape[3] == 28, x_test.shape

    assert y_test.ndim == 1, y_test.ndim
    assert y_test.shape[0] == 10000, y_test.shape


def test_load_subset_mnist():

    (x_train, y_train, x_test, y_test) = get_full_batch_data("mnist", \
                                "~/data",
                                 subset_percentage=0.001)
    
    n_selected = x_train.shape[0]
    logger.info(f"Selected {round(100. * n_selected / 60000)}% of MNIST training set")
    
    assert x_train.ndim == 4, x_train.ndim
    assert x_train.shape[0] == n_selected, x_train.shape
    assert x_train.shape[1] == 1, x_train.shape
    assert x_train.shape[2] == 28, x_train.shape
    assert x_train.shape[3] == 28, x_train.shape

    assert y_train.ndim == 1, y_train.ndim
    assert y_train.shape[0] == n_selected, y_train.shape

    n_selected_test = x_test.shape[0]
    logger.info(f"Selected {round(100. * n_selected_test / 10000)}% of MNIST validation set")

    assert x_test.ndim == 4, x_test.ndim
    assert x_test.shape[0] == n_selected_test, x_test.shape
    assert x_test.shape[1] == 1, x_test.shape
    assert x_test.shape[2] == 28, x_test.shape
    assert x_test.shape[3] == 28, x_test.shape

    assert y_test.ndim == 1, y_test.ndim
    assert y_test.shape[0] == n_selected_test, y_test.shape

    
    

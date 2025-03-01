import fire
import numpy as np
import torch
from loguru import logger
from torch.utils.data import Subset
from torchvision import datasets, transforms


class DataOptions:

    def __init__(self, dataset, path, bs_train, bs_eval, resize=None, class_list=None):
        self.dataset = dataset
        self.path = path
        self.batch_size_train = bs_train
        self.batch_size_eval = bs_eval
        self.resize = resize
        self.class_list = class_list


def get_data_simple(dataset, path, bs_train, bs_eval, subset=None, resize=None, class_list=None):

    return get_data(DataOptions(
        dataset,
        path,
        bs_train,
        bs_eval,
        resize,
        class_list
    ), subset_percentage=subset)


@torch.no_grad()
def recover_eval_tensors(dataloader):

    final_x, final_y = [], []

    for x, y in dataloader:

        final_x.append(x)
        final_y.append(y)

    return torch.cat(final_x, 0), torch.cat(final_y, 0)


def get_full_batch_data(dataset: str,
                         path: str,
                          subset_percentage: float = 0.01,
                          resize: float = 28,
                          class_list=None):

    train_loader, test_loader, *_ = get_data(DataOptions(
        dataset,
        path,
        100,
        100,
        resize,
        class_list
    ), subset_percentage=subset_percentage)

    x_train, y_train = recover_eval_tensors(train_loader)
    x_test, y_test = recover_eval_tensors(test_loader)

    logger.debug(f'{x_train.shape}')
    logger.debug(f'{y_train.shape}')
    logger.debug(f'{x_test.shape}')
    logger.debug(f'{y_test.shape}')

    return (x_train, y_train, x_test, y_test)



def get_data(args: DataOptions, subset_percentage: float = None):

    # mean/std stats
    if args.dataset == 'cifar10':
        data_class = 'CIFAR10'
        num_classes = 10
        stats = {
            'mean': [0.491, 0.482, 0.447],
            'std': [0.247, 0.243, 0.262]
        }
    elif args.dataset == 'cifar100':
        data_class = 'CIFAR100'
        num_classes = 100
        stats = {
            'mean': [0.5071, 0.4867, 0.4408],
            'std': [0.2675, 0.2565, 0.2761]
        }
    elif args.dataset == 'mnist':
        data_class = 'MNIST'
        num_classes = 10
        stats = {
            'mean': [0.1307],
            'std': [0.3081]
        }
    elif args.dataset == "fashion-mnist":
        data_class = 'FashionMNIST'
        num_classes = 10
        stats = {
            'mean': [0.1307],
            'std': [0.3081]
        }
    else:
        raise ValueError("unknown dataset")

    # input transformation w/o preprocessing for now

    trans = [
        transforms.ToTensor(),
        lambda t: t.type(torch.get_default_dtype()),
        transforms.Normalize(**stats)
    ]

    if args.dataset == "mnist" and args.resize is not None:
        trans = [
            transforms.ToTensor(),
            lambda t: t.type(torch.get_default_dtype()),
            transforms.Normalize(**stats),
            transforms.Resize(args.resize)
        ]

    if args.dataset == "fashion-mnist" and args.resize is not None:
        trans = [
            transforms.ToTensor(),
            lambda t: t.type(torch.get_default_dtype()),
            transforms.Normalize(**stats),
            transforms.Resize(args.resize)
        ]

    # get train and test data with the same normalization
    tr_data = getattr(datasets, data_class)(
        root=args.path,
        train=True,
        download=True,
        transform=transforms.Compose(trans)
    )

    te_data = getattr(datasets, data_class)(
        root=args.path,
        train=False,
        download=True,
        transform=transforms.Compose(trans)
    )

    n_tr = len(tr_data)
    n_te = len(te_data)

    if args.class_list is None:
        class_list_keys = tr_data.class_to_idx.keys()
        class_list = [tr_data.class_to_idx[k] for k in class_list_keys]
        # TODO: improve this
        # we assue that the classes of train ad test are the same
    else:
        class_list = args.class_list
    logger.info(f"Used classes: {class_list}")

    # TODO: fix this subset 1 issue
    if subset_percentage >= 1.:
        logger.warning("subset 1 may be unstable")

    # We try to extract subsets equivalently in each class to keep them balanced
    # Subset selection is performed only on the training set!!

    assert subset_percentage > 0. and subset_percentage <= 1.
    logger.info(f"Using {round(100. * subset_percentage, 2)}% of the {data_class} training set")

    selected_indices = torch.zeros(len(tr_data), dtype=torch.bool)
    for cl_idx in class_list:
        where_class = torch.where(torch.tensor(tr_data.targets) == cl_idx)[0]
        sub_indices = (torch.rand(len(where_class)) < subset_percentage)
        selected_indices[where_class[sub_indices]] = True

    tr_data = Subset(tr_data, selected_indices.nonzero().reshape(-1))

    # TODO: this is hacky amd random, improve it  HACK
    subset_eval = min(1., 5. * subset_percentage)

    # We try to extract subsets equivalently in each class to keep them balanced
    # Subset selection is performed only on the training set!!

    assert subset_eval > 0. and subset_eval <= 1.
    logger.info(f"Using only {round(100. * subset_eval, 2)}% of the {data_class} validation set")

    selected_indices = torch.zeros(len(te_data), dtype=torch.bool)
    for cl_idx in class_list:
        where_class = torch.where(torch.tensor(te_data.targets) == cl_idx)[0]
        sub_indices = (torch.rand(len(where_class)) < subset_eval)
        selected_indices[where_class[sub_indices]] = True

    te_data = Subset(te_data, selected_indices.nonzero().reshape(-1))

    # get tr_loader for train/eval and te_loader for eval
    train_loader = torch.utils.data.DataLoader(
        dataset=tr_data,
        batch_size=args.batch_size_train,
        shuffle=True,
    )

    train_loader_eval = torch.utils.data.DataLoader(
        dataset=tr_data,
        batch_size=args.batch_size_eval,
        shuffle=False,
    )

    test_loader_eval = torch.utils.data.DataLoader(
        dataset=te_data,
        batch_size=args.batch_size_eval,
        shuffle=False,
    )

    return train_loader, test_loader_eval, train_loader_eval, num_classes


if __name__ == "__main__":

    def test_function():
        _ = get_data_simple("mnist", "~/data/", 1, 1, subset=1., class_list = [1, 7])       

    fire.Fire(test_function)
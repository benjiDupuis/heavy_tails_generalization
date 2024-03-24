import json
from pathlib import Path

import fire
import numpy as np
import torch
from loguru import logger

from data.dataset import get_data_simple
from last_point.eval import eval
from last_point.model import NoisyCNN
from last_point.utils import accuracy


def pretrain(n_iter=10000,\
            log_freq=1000,
            width=120, 
            bs=128,
            subset=1.,
            result_dir="weights",
            epochs=2):

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"On device {device}")

    data = get_data_simple("cifar10", "~/data", bs, 1000, subset=subset)

    # data
    train_loader = data[0]
    test_loader_eval = data[1]
    train_loader_eval = data[2]

    model = NoisyCNN(width=width)
    logger.info(f"Used model: {model}")
    model.to(device)
    print(model)

    opt = torch.optim.Adam(model.parameters(),
                           lr = eta)
    crit = nn.CrossEntropyLoss().to(device)

    loss_tab = []
    batch_loss_tab = []

    for epoch in range(epochs):

        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(trainloader, 0):
        
            inputs, labels = data

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = crit(outputs, labels)
            loss.backward()
            optimizer.step()

            acc = accuracy(outputs, labels)

            # print statistics
            running_loss += loss.item()
            running_acc += acc
            if i // log_freq == 0:

                logger.info(f'[{epoch + 1}, {i + 1}] loss: {round(running_loss / log_freq), 2} accuracy: {round(running_acc / log_freq), 2}')
                running_loss = 0.0
                running_acc = 0.0

    # Validation at the end
    with torch.no_grad():
        te_hist, *_ = eval(test_loader_eval, model, crit_unreduced, opt)
        tr_hist, *_ = eval(train_loader_eval, model, crit_unreduced, opt)

    acc_train = tr_hst[1]
    acc_test = te_hst[1]

    logger.info(f"Final train accuracy: {100. * round(acc_train, 2)} %")
    logger.info(f"Final test accuracy: {100. * round(acc_test, 2)} %")

    # Saving and logging
    train_dir = Path(result_dir) / str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").split(".")[0]

    if not train_dir.is_dir():
        train_dir.mkdir(exist_ok=True, parents=True)

    json_path = train_dir / "train_log.json"
    weight_path = train_dir / "weights.pth"

    exp_dict = {
        "epochs" : epochs,
        "batch_size": bs,
        "subset": subset,
        "train_accuacy": acc_train
        "test_accuacy": acc_test
        "width": width,
        "model": model.__str__(),
        "weights": str(weight_path)
    }

    logger.info(f"Results: {json.dumps(exp_dict, indent=2)}")

    logger.info(f"Saving logs in {str(json_path)}")
    with open(str(json_path), "w") as json_file:
        json.dump(exp_dict, json_file, indent=2)

    logger.info(f"Saving logs in {json.dumps(exp_dict, indent=2)}")
    torch.save(model.state_dict(), str(weight_path))


if __name__ == "__main__":
    fire.Fire(pretrain)














import argparse
import torch
import wandb
from tqdm import tqdm
import yaml
from dataloaders import SequenceDataset

from models import Mamba, Hawk

def train_mamba(seed, trainloader, testloader, train_config, model_config):
    torch.manual_seed(seed)
    device = "cuda"
    model = Mamba(**model_config).to(device)
    nr_params = sum(p.numel() for p in model.parameters())
    print("Nr. of parameters: {0}".format(nr_params))
    wandb.log({"params": nr_params})
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["lr"], weight_decay=train_config["wd"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config["num_epochs"], eta_min = 5e-6)
    running_loss = 0.0
    for epoch in range(train_config["num_epochs"]):
        for X, y, _ in tqdm(trainloader):
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            y_hat = model(X)
            loss = torch.nn.functional.cross_entropy(y_hat, y)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = running_loss/len(trainloader)
        print("Loss: {0:.3f}".format(train_loss))
        scheduler.step()

        model.eval()
        running_accuracy = 0.0
        with torch.no_grad():
            for X, y, _ in tqdm(trainloader):
                X = X.to(device)
                y = y.to(device)
                y_hat = model(X)
                accuracy = (y_hat.argmax(dim=1) == y).float().sum() / len(y)
                running_accuracy += accuracy
        train_acc = running_accuracy / len(trainloader)
        print("Train accuracy: {0:.4f}".format(train_acc))

        running_accuracy = 0.0
        running_loss = 0.0
        with torch.no_grad():
            for X, y, _ in tqdm(testloader):
                X = X.to(device)
                y = y.to(device)
                y_hat = model(X)
                loss = torch.nn.functional.cross_entropy(y_hat, y)
                running_loss += loss.item()
                accuracy = (y_hat.argmax(dim=1) == y).float().sum() / len(y)
                running_accuracy += accuracy
        test_loss = running_loss/len(testloader)
        test_acc = running_accuracy / len(testloader)
        print("Test accuracy: {0:.4f}\n".format(test_acc))

        wandb.log({"train acc": train_acc,
                   "test acc": test_acc,
                   "train loss": train_loss,
                   "test loss": test_loss,
                   "lr": optimizer.param_groups[0]['lr']})
        model.train()

def train_hawk(seed, trainloader, testloader, train_config, model_config):
    torch.manual_seed(seed)
    device = "cuda"
    model = Hawk(**model_config).to(device)
    nr_params = sum(p.numel() for p in model.parameters())
    print("Nr. of parameters: {0}".format(nr_params))
    wandb.log({"params": nr_params})
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["lr"], weight_decay=train_config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config["num_epochs"], eta_min = 5e-6)
    running_loss = 0.0
    for epoch in range(train_config["num_epochs"]):
        for X, y, _ in tqdm(trainloader):
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            y_hat = model(X)
            loss = torch.nn.functional.cross_entropy(y_hat, y)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = running_loss/len(trainloader)
        print("Loss: {0:.3f}".format(train_loss))
        scheduler.step()

        model.eval()
        running_accuracy = 0.0
        with torch.no_grad():
            for X, y, _ in tqdm(trainloader):
                X = X.to(device)
                y = y.to(device)
                y_hat = model(X)
                accuracy = (y_hat.argmax(dim=1) == y).float().sum() / len(y)
                running_accuracy += accuracy
        train_acc = running_accuracy / len(trainloader)
        print("Train accuracy: {0:.4f}".format(train_acc))

        running_accuracy = 0.0
        running_loss = 0.0
        with torch.no_grad():
            for X, y, _ in tqdm(testloader):
                X = X.to(device)
                y = y.to(device)
                y_hat = model(X)
                loss = torch.nn.functional.cross_entropy(y_hat, y)
                running_loss += loss.item()
                accuracy = (y_hat.argmax(dim=1) == y).float().sum() / len(y)
                running_accuracy += accuracy
        test_loss = running_loss/len(testloader)
        test_acc = running_accuracy / len(testloader)
        print("Test accuracy: {0:.4f}\n".format(test_acc))

        wandb.log({"train acc": train_acc,
                   "test acc": test_acc,
                   "train loss": train_loss,
                   "test loss": test_loss,
                   "lr": optimizer.param_groups[0]['lr']})
        model.train()

def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0-val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(42),
    )
    return train, val

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="cifar-10.yaml", help="experiment config file")
    config = parser.parse_args().config
    print("\nUsing config {0}".format(config))

    # get GPU info
    if not torch.cuda.is_available():
        raise NotImplementedError("Cannot run on CPU!")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_type = torch.cuda.get_device_name(0)
    print("Running on {0}".format(gpu_type))
    
    # get args
    with open("configs/"+config) as stream:
        try:
            args = yaml.safe_load(stream)            
        except yaml.YAMLError as exc:
            raise RuntimeError(exc)

    args["GPU"] = gpu_type
    
    print("\nCONFIG:")
    print(yaml.dump(args))

    ## split configs
    data_config = args["dataset"]
    train_config = args["train"]
    model_config = args["model"]
    layer = model_config.pop("layer") # remove layer name

    # start wandb logging
    wandb.login(key="58d1b0b4e77ad3dd9ebd08eb490255e83aa70bfe")
    wandb.init(
            entity="ssm-eth",
            project="lra-benchmark",
            config=args,
            job_type="train",
    )
    
    ## prepare dataset
    data_config.pop("name") # remove logging name
    dataset = SequenceDataset.registry[data_config["_name_"]](**data_config)
    dataset.setup()

    # Dataloaders
    trainloader = dataset.train_dataloader(batch_size=train_config["batch_size"], shuffle=True)
    testloader = dataset.test_dataloader(batch_size=train_config["batch_size"], shuffle=False)
    if type(testloader) is dict:
        testloader = testloader[None]
    
    if layer == "mamba":
        train_mamba(
            args["seed"],
            trainloader,
            testloader,
            train_config,
            model_config
        )
    elif layer == "hawk":
        train_hawk(
            args["seed"],
            trainloader,
            testloader,
            train_config,
            model_config
        )
    else:
        raise RuntimeError("{0} is not a valid model option".format(layer))

    wandb.finish()
